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
    def test_a_policy_class_with_run_is_loaded(self, tmp_path):
        root = write_policy(
            tmp_path / "p",
            "class Policy:\n    def run(self, questions, ctx): return 'whole set'\n",
        )
        assert discover_policy(root).run([], None) == "whole set"

    def test_a_module_function_is_a_policy(self, tmp_path):
        root = write_policy(tmp_path / "p", "def run(questions, ctx):\n    return 'm'\n")
        assert discover_policy(root).run([], None) == "m"

    def test_solve_alone_is_no_longer_a_policy(self, tmp_path):
        root = write_policy(tmp_path / "p", "def solve(question, ctx):\n    return 'A'\n")
        with pytest.raises(PolicyError, match=r"must define run\(questions, ctx\)"):
            discover_policy(root)

    def test_a_module_function_with_the_wrong_signature_is_rejected(self, tmp_path):
        root = write_policy(
            tmp_path / "p",
            "def run(questions, model_client, context):\n    return None\n",
        )
        with pytest.raises(PolicyError, match=r"run\(\) must take exactly"):
            discover_policy(root)

    def test_policy_instance_is_accepted(self, tmp_path):
        root = write_policy(
            tmp_path / "p",
            "class _P:\n    def run(self, qs, ctx): return 'A'\npolicy = _P()\n",
        )
        assert discover_policy(root).run([], None) == "A"

    def test_helper_modules_are_importable(self, tmp_path):
        root = write_policy(
            tmp_path / "p",
            "from helpers import pick\nclass Policy:\n    def run(self, qs, ctx): return pick()\n",
            helpers__py="def pick(): return 'C'\n",
        )
        assert discover_policy(root).run([], None) == "C"

    def test_module_is_imported_exactly_once(self, tmp_path):
        """Import-time side effects must happen once."""
        root = write_policy(
            tmp_path / "p",
            "import json\n"
            "COUNT = []\n"
            "COUNT.append(1)\n"
            "class Policy:\n"
            "    def run(self, qs, ctx): return str(len(COUNT))\n",
        )
        assert discover_policy(root).run([], None) == "1"

    def test_missing_run_reports_the_contract(self, tmp_path):
        root = write_policy(tmp_path / "p", "answer = 1\n")
        with pytest.raises(PolicyError, match=r"must define run\(questions, ctx\)"):
            discover_policy(root)

    def test_wrong_run_arity_is_rejected(self, tmp_path):
        root = write_policy(
            tmp_path / "p", "class Policy:\n    def run(self, questions): return {}\n"
        )
        with pytest.raises(PolicyError, match="exactly \\(questions, ctx\\)"):
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
        assert manifest.name == "policy"
        assert manifest.max_tokens_per_call == 8192

    @pytest.mark.parametrize(
        "key", ["sequential", "memory", "concurrent", "setup_calls", "components"]
    )
    def test_unknown_or_removed_keys_are_errors_not_silent_noops(self, key):
        with pytest.raises(PolicyError, match="unknown key"):
            manifest_from_mapping({key: True})

    def test_object_manifest_beats_module_manifest(self, tmp_path):
        root = write_policy(
            tmp_path / "p",
            "MANIFEST = {'name': 'module'}\n"
            "class Policy:\n"
            "    MANIFEST = {'name': 'object'}\n"
            "    def run(self, qs, ctx): return None\n",
        )
        assert discover_policy(root).manifest.name == "object"

    def test_bad_numbers_are_rejected(self):
        with pytest.raises(ValueError, match="max_tokens_per_call"):
            PolicyManifest(max_tokens_per_call=0)
