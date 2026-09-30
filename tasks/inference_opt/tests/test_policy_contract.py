"""Tests for policy loading and validation."""

from __future__ import annotations

import pytest
from inference_opt.api import PolicyManifest
from inference_opt.policy import PolicyError, discover_policy, manifest_from_mapping


def write_policy(root, source, name="policy.py", **extra):
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(source, encoding="utf-8")
    for filename, body in extra.items():
        (root / filename.replace("__", ".")).write_text(body, encoding="utf-8")
    return root


class TestDiscovery:
    def test_class_policy_with_setup(self, tmp_path):
        root = write_policy(
            tmp_path / "p",
            "class Policy:\n"
            "    def setup(self, ctx): pass\n"
            "    def solve(self, q, ctx): return 'A'\n",
        )
        loaded = discover_policy(root)
        assert loaded.has_setup
        assert not loaded.is_legacy

    def test_legacy_module_function_is_wrapped(self, tmp_path):
        root = write_policy(
            tmp_path / "p",
            "def solve(question, model_client, context):\n    return question.upper()\n",
        )
        loaded = discover_policy(root)
        assert loaded.is_legacy
        assert not loaded.has_setup

    def test_policy_instance_is_accepted(self, tmp_path):
        root = write_policy(
            tmp_path / "p",
            "class _P:\n    def solve(self, q, ctx): return 'A'\npolicy = _P()\n",
        )
        assert discover_policy(root).obj.solve(None, None) == "A"

    def test_helper_modules_are_importable(self, tmp_path):
        root = write_policy(
            tmp_path / "p",
            "from helpers import pick\nclass Policy:\n    def solve(self, q, ctx): return pick()\n",
            helpers__py="def pick(): return 'C'\n",
        )
        assert discover_policy(root).obj.solve(None, None) == "C"

    def test_module_is_imported_exactly_once(self, tmp_path):
        """The original implementation loaded policy.py twice, running side effects twice."""
        root = write_policy(
            tmp_path / "p",
            "import json\n"
            "COUNT = []\n"
            "COUNT.append(1)\n"
            "class Policy:\n"
            "    def solve(self, q, ctx): return str(len(COUNT))\n",
        )
        assert discover_policy(root).obj.solve(None, None) == "1"

    def test_missing_solve_reports_the_contract(self, tmp_path):
        root = write_policy(tmp_path / "p", "answer = 1\n")
        with pytest.raises(PolicyError, match="callable solve"):
            discover_policy(root)

    def test_wrong_arity_is_rejected(self, tmp_path):
        root = write_policy(
            tmp_path / "p", "class Policy:\n    def solve(self, q): return ''\n"
        )
        with pytest.raises(PolicyError, match="exactly \\(question, ctx\\)"):
            discover_policy(root)

    def test_import_time_failure_is_reported_not_swallowed(self, tmp_path):
        root = write_policy(tmp_path / "p", "raise ValueError('boom')\n")
        with pytest.raises(PolicyError, match="ValueError.*boom"):
            discover_policy(root)

    def test_missing_directory(self, tmp_path):
        with pytest.raises(PolicyError, match="does not exist"):
            discover_policy(tmp_path / "nope")


class TestManifest:
    def test_defaults(self):
        manifest = manifest_from_mapping(None)
        assert manifest.concurrent is True
        assert manifest.max_tokens_per_call == 8192

    def test_unknown_key_is_an_error_not_a_silent_noop(self):
        # `concurrent` is the only execution knob; look-alikes fail loudly.
        with pytest.raises(PolicyError, match="unknown key"):
            manifest_from_mapping({"sequential": True})

    def test_memory_is_no_longer_a_manifest_key(self):
        with pytest.raises(PolicyError, match="unknown key"):
            manifest_from_mapping({"memory": "shared"})

    def test_concurrent_can_be_disabled(self):
        assert manifest_from_mapping({"concurrent": False}).concurrent is False

    def test_concurrent_must_be_a_bool(self):
        with pytest.raises(PolicyError, match="concurrent"):
            manifest_from_mapping({"concurrent": "disable"})

    def test_components_accept_strings_and_dicts(self):
        manifest = manifest_from_mapping(
            {"components": ["critic", {"name": "proposer", "kind": "sampler"}]}
        )
        assert [component.name for component in manifest.components] == [
            "critic",
            "proposer",
        ]
        assert manifest.components[1].kind == "sampler"

    def test_invalid_execution_value(self):
        with pytest.raises(PolicyError, match="unknown key"):
            manifest_from_mapping({"execution": "whenever"})

    def test_object_manifest_beats_module_manifest(self, tmp_path):
        root = write_policy(
            tmp_path / "p",
            "MANIFEST = {'name': 'module'}\n"
            "class Policy:\n"
            "    MANIFEST = {'name': 'object'}\n"
            "    def solve(self, q, ctx): return ''\n",
        )
        assert discover_policy(root).manifest.name == "object"

    def test_bad_numbers_are_rejected(self):
        with pytest.raises(ValueError, match="max_calls_per_question"):
            PolicyManifest(max_calls_per_question=0)
