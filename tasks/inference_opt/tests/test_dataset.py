"""The shipped frozen question set.

These run against the committed dataset, not a fixture: if it is rebuilt and a
split changes size, a target leaks into the public files, or the splits stop
being equally hard, every score measured afterwards is silently wrong.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter

import pytest
from inference_opt import datasets

SPLITS = ("train", "test", "examples")
MATH = [b for b in datasets.BENCHMARKS if b.startswith("math_")]
GSM = [b for b in datasets.BENCHMARKS if b.startswith("gsm_")]
MATH_SIZES = {"train": 30, "test": 30, "examples": 20}
#: Templates per split; every template has five instances.
GSM_TEMPLATES = {
    "gsm_symbolic": {"train": 30, "test": 30, "examples": 20},
    "gsm_p1": {"train": 30, "test": 30, "examples": 20},
    "gsm_p2": {"train": 20, "test": 20, "examples": 10},
}


@pytest.fixture(scope="module")
def manifest():
    return datasets.load_manifest()


class TestShippedDataset:
    def test_manifest_records_the_protocol(self, manifest):
        assert manifest["dataset_version"] == datasets.DATASET_VERSION
        assert manifest["benchmarks"] == list(datasets.BENCHMARKS)
        assert len(manifest["content_fingerprint"]) == 64
        assert set(manifest["selection"]) == {"math", "gsm"}

    @pytest.mark.parametrize("benchmark", datasets.BENCHMARKS)
    def test_splits_do_not_overlap(self, benchmark):
        items = datasets.load_items(benchmark)
        assert len({item.item_id for item in items}) == len(items)
        assert len({item.question_hash for item in items}) == len(items)

    @pytest.mark.parametrize("benchmark", datasets.BENCHMARKS)
    def test_every_item_has_a_label(self, benchmark):
        for split in SPLITS:
            targets = datasets.load_targets(benchmark, split)
            missing = [
                item.item_id
                for item in datasets.load_items(benchmark, split)
                if item.item_id not in targets
            ]
            assert not missing, f"{benchmark}/{split} missing labels: {missing[:3]}"

    @pytest.mark.parametrize("benchmark", datasets.BENCHMARKS)
    def test_public_files_never_contain_an_answer(self, benchmark):
        """The boundary the whole environment rests on."""
        path = datasets.data_root() / "public" / f"{benchmark}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            assert "target" not in record
            assert "answer" not in record

    def test_identifiers_trace_back_to_the_source_dataset(self):
        for benchmark in datasets.BENCHMARKS:
            for item in datasets.load_items(benchmark)[:5]:
                assert item.item_id == f"{benchmark}:{item.sample_id}"


class TestMath:
    @pytest.mark.parametrize("benchmark", MATH)
    def test_splits_have_their_size(self, benchmark):
        sizes = {split: len(datasets.load_items(benchmark, split)) for split in SPLITS}
        assert sizes == MATH_SIZES

    @pytest.mark.parametrize("benchmark", MATH)
    def test_each_benchmark_is_one_level_5_subject(self, benchmark, manifest):
        items = datasets.load_items(benchmark)
        assert {item.category for item in items} == {manifest["per_benchmark"][benchmark]["subject"]}
        assert {item.subcategory for item in items} == {"level 5"}
        assert all(item.sample_id.startswith("math_") for item in items)

    @pytest.mark.parametrize("benchmark", MATH)
    def test_the_splits_are_equally_hard(self, benchmark, manifest):
        entry = manifest["per_benchmark"][benchmark]
        means = [entry[f"{split}_mean_irt_difficulty"] for split in SPLITS]
        assert max(means) - min(means) < 0.1, f"{benchmark}: split difficulty {means}"

    def test_answer_format_never_says_whether_the_answer_is_a_number(self):
        """A per-question numeric/expression label would hint at the answer."""
        formats = {item.answer_format for b in MATH for item in datasets.load_items(b)}
        assert formats == {"math"}

    def test_every_gold_answer_is_gradable_by_inspect_evals(self):
        """Questions whose gold answer sympy cannot match to itself were excluded."""
        from inspect_evals.math.utils import is_equiv_sympy, normalize_final_answer

        async def gradable(target: str) -> bool:
            normalized = await normalize_final_answer(target)
            return bool(await is_equiv_sympy(normalized, normalized))

        targets = [t for b in MATH for t in datasets.load_targets(b).values()]
        failing = [t for t in targets if not asyncio.run(gradable(t))]
        assert not failing, failing[:5]


class TestGsmSymbolic:
    @pytest.mark.parametrize("benchmark", GSM)
    def test_every_template_has_five_instances_in_one_split(self, benchmark):
        templates: dict[str, set[str]] = {}
        counts: Counter[str] = Counter()
        for item in datasets.load_items(benchmark):
            templates.setdefault(item.category, set()).add(item.split)
            counts[item.category] += 1
        assert all(len(splits) == 1 for splits in templates.values())
        assert set(counts.values()) == {5}
        per_split = Counter(next(iter(splits)) for splits in templates.values())
        assert dict(per_split) == GSM_TEMPLATES[benchmark]

    def test_a_gsm8k_question_is_in_the_same_split_in_every_variant(self):
        """No test problem is seen in train or the examples of another variant."""
        split_of: dict[str, set[str]] = {}
        for benchmark in GSM:
            for item in datasets.load_items(benchmark):
                split_of.setdefault(item.subcategory, set()).add(item.split)
        assert all(len(splits) == 1 for splits in split_of.values())

    @pytest.mark.parametrize("benchmark", GSM)
    def test_ids_name_their_template(self, benchmark):
        from inference_opt.score import template_of

        for item in datasets.load_items(benchmark):
            template, _, instance = item.sample_id.partition("-")
            assert item.category == f"template {template}"
            assert template_of(item.item_id) == f"{benchmark}:{template}"
            assert instance.isdigit()

    def test_every_gold_answer_is_a_number(self):
        targets = [t for b in GSM for t in datasets.load_targets(b).values()]
        assert all(float(t) == float(t) for t in targets)
        formats = {item.answer_format for b in GSM for item in datasets.load_items(b)}
        assert formats == {"numeric"}


class TestPolicyFacingQuestions:
    def test_public_records_become_questions(self):
        from inference_opt.host import _question

        items = datasets.load_items("math_algebra", "test")
        records = [datasets.public_record(item) for item in items]
        questions = [_question(record, i, len(records)) for i, record in enumerate(records)]
        assert [question.index for question in questions] == list(range(len(questions)))
        assert all(question.total == MATH_SIZES["test"] for question in questions)
        assert {question.answer_type for question in questions} == {"text"}
        assert all(question.choices is None for question in questions)

    def test_gsm_questions_name_their_template_as_topic(self):
        from inference_opt.host import _question

        item = datasets.load_items("gsm_p1", "train")[0]
        question = _question(datasets.public_record(item), 0, 1)
        assert question.topic == item.category
        assert question.answer_type == "numeric"

    def test_public_record_never_carries_the_target(self):
        item = datasets.load_items("math_geometry", "test")[0]
        record = datasets.public_record(item)
        assert "target" not in record
        assert record["item_id"] == item.item_id
