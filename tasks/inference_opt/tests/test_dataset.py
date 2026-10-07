"""The shipped frozen question set.

These run against the committed dataset, not a fixture: if it is rebuilt and a
split changes size, a target leaks into the public files, or the splits stop
being equally hard, every score measured afterwards is silently wrong.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from inference_opt import datasets

SIZES = {"train": 30, "test": 30, "examples": 20}


@pytest.fixture(scope="module")
def manifest():
    return datasets.load_manifest()


class TestShippedDataset:
    def test_manifest_records_the_protocol(self, manifest):
        assert manifest["dataset_version"] == datasets.DATASET_VERSION
        assert manifest["benchmarks"] == list(datasets.BENCHMARKS)
        assert len(manifest["content_fingerprint"]) == 64
        assert manifest["train_per_benchmark"] == SIZES["train"]
        assert manifest["test_per_benchmark"] == SIZES["test"]
        assert manifest["examples_per_benchmark"] == SIZES["examples"]

    @pytest.mark.parametrize("benchmark", datasets.BENCHMARKS)
    def test_splits_have_their_size_and_do_not_overlap(self, benchmark):
        splits = {split: datasets.load_items(benchmark, split) for split in SIZES}
        assert {split: len(items) for split, items in splits.items()} == SIZES
        ids = [item.item_id for items in splits.values() for item in items]
        hashes = [item.question_hash for items in splits.values() for item in items]
        assert len(set(ids)) == len(ids)
        assert len(set(hashes)) == len(hashes)

    @pytest.mark.parametrize("benchmark", datasets.BENCHMARKS)
    def test_each_benchmark_is_one_level_5_subject(self, benchmark, manifest):
        items = datasets.load_items(benchmark)
        assert {item.category for item in items} == {manifest["per_benchmark"][benchmark]["subject"]}
        assert {item.subcategory for item in items} == {"level 5"}

    @pytest.mark.parametrize("benchmark", datasets.BENCHMARKS)
    def test_the_splits_are_equally_hard(self, benchmark, manifest):
        entry = manifest["per_benchmark"][benchmark]
        means = [entry[f"{split}_mean_irt_difficulty"] for split in SIZES]
        assert max(means) - min(means) < 0.1, f"{benchmark}: split difficulty {means}"

    @pytest.mark.parametrize("benchmark", datasets.BENCHMARKS)
    def test_every_item_has_a_label(self, benchmark):
        for split in SIZES:
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

    def test_answer_format_never_says_whether_the_answer_is_a_number(self):
        """A per-question numeric/expression label would hint at the answer."""
        formats = {item.answer_format for b in datasets.BENCHMARKS for item in datasets.load_items(b)}
        assert formats == {"math"}

    def test_every_gold_answer_is_gradable_by_inspect_evals(self):
        """Questions whose gold answer sympy cannot match to itself were excluded."""
        from inspect_evals.math.utils import is_equiv_sympy, normalize_final_answer

        async def gradable(target: str) -> bool:
            normalized = await normalize_final_answer(target)
            return bool(await is_equiv_sympy(normalized, normalized))

        targets = [t for b in datasets.BENCHMARKS for t in datasets.load_targets(b).values()]
        failing = [t for t in targets if not asyncio.run(gradable(t))]
        assert not failing, failing[:5]

    def test_identifiers_trace_back_to_the_source_dataset(self):
        for benchmark in datasets.BENCHMARKS:
            for item in datasets.load_items(benchmark)[:5]:
                assert item.item_id == f"{benchmark}:{item.sample_id}"
                assert item.sample_id.startswith("math_")


class TestPolicyFacingQuestions:
    def test_public_records_become_questions(self):
        from inference_opt.host import _question

        items = datasets.load_items("math_algebra", "test")
        records = [datasets.public_record(item) for item in items]
        questions = [_question(record, i, len(records)) for i, record in enumerate(records)]
        assert [question.index for question in questions] == list(range(len(questions)))
        assert all(question.total == SIZES["test"] for question in questions)
        assert {question.answer_type for question in questions} == {"text"}
        assert all(question.choices is None for question in questions)

    def test_public_record_never_carries_the_target(self):
        item = datasets.load_items("math_geometry", "test")[0]
        record = datasets.public_record(item)
        assert "target" not in record
        assert record["item_id"] == item.item_id
