"""The shipped frozen question set.

These run against the committed dataset, not a fixture: if it is ever rebuilt and
the split sizes change, a target leaks into the public files, or the halves stop
being matched, that silently corrupts every score measured afterwards.
"""

from __future__ import annotations

import json

import pytest
from inference_opt import datasets

TRAIN_PER_BENCHMARK = 30
TEST_PER_BENCHMARK = 30
#: The band every selected item was required to fall in.
BAND = (0.20, 0.80)


@pytest.fixture(scope="module")
def manifest():
    return datasets.load_manifest()


class TestShippedDataset:
    def test_manifest_records_the_protocol(self, manifest):
        assert manifest["dataset_version"] == datasets.DATASET_VERSION
        assert len(manifest["content_fingerprint"]) == 64
        assert manifest["items_per_benchmark"] == 60
        assert manifest["selection"]["band"] == list(BAND)

    @pytest.mark.parametrize("benchmark", datasets.BENCHMARKS)
    def test_every_benchmark_has_a_matched_split(self, benchmark):
        train = datasets.load_items(benchmark, "train")
        test = datasets.load_items(benchmark, "test")
        assert len(train) == TRAIN_PER_BENCHMARK
        assert len(test) == TEST_PER_BENCHMARK
        assert not {item.item_id for item in train} & {item.item_id for item in test}

    @pytest.mark.parametrize("benchmark", datasets.BENCHMARKS)
    def test_the_halves_are_equally_hard(self, benchmark, manifest):
        entry = manifest["per_benchmark"][benchmark]
        gap = abs(
            entry["train_reference_accuracy"] - entry["test_reference_accuracy"]
        )
        assert gap < 0.15, f"{benchmark}: halves differ by {gap:.3f}"

    @pytest.mark.parametrize("benchmark", datasets.BENCHMARKS)
    def test_every_item_sits_in_the_movable_band(self, benchmark):
        """Outside this band no test-time strategy could shift the outcome."""
        for item in datasets.load_items(benchmark):
            assert item.reference_accuracy is not None
            assert BAND[0] <= item.reference_accuracy <= BAND[1], item.item_id

    @pytest.mark.parametrize("benchmark", datasets.BENCHMARKS)
    def test_every_item_has_a_label(self, benchmark):
        for split in ("train", "test"):
            items = datasets.load_items(benchmark, split)
            targets = datasets.load_targets(benchmark, split)
            missing = [item.item_id for item in items if item.item_id not in targets]
            assert not missing, f"{benchmark}/{split} missing labels: {missing[:3]}"

    @pytest.mark.parametrize("benchmark", datasets.BENCHMARKS)
    def test_public_files_never_contain_an_answer(self, benchmark):
        """The boundary the whole environment rests on."""
        path = datasets.data_root() / "public" / f"{benchmark}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                assert "target" not in record
                assert "answer" not in record

    @pytest.mark.parametrize("benchmark", datasets.BENCHMARKS)
    def test_multiple_choice_items_carry_their_options(self, benchmark):
        for item in datasets.load_items(benchmark):
            if item.answer_format.startswith("mcq"):
                assert item.options, f"{item.item_id} is mcq with no options"

    def test_identifiers_trace_back_to_the_source_dataset(self):
        for benchmark in datasets.BENCHMARKS:
            for item in datasets.load_items(benchmark)[:5]:
                assert item.item_id == f"{benchmark}:{item.sample_id}"

    def test_bbh_questions_are_stripped_of_their_few_shot_scaffold(self):
        """Source bbh rows ship as pre-rendered 3-shot prompts, not bare questions."""
        for item in datasets.load_items("bbh"):
            assert "QUESTION:" not in item.question

    def test_chembench_keeps_its_answer_formats(self):
        formats = {item.answer_format for item in datasets.load_items("chembench")}
        assert "mcq_single" in formats
        assert formats <= {"mcq_single", "mcq_multi", "numeric"}

    def test_categories_are_spread_not_clustered(self):
        """A set drawn from two of bbh's subtasks would measure almost nothing."""
        categories = {
            item.category for item in datasets.load_items("bbh") if item.category
        }
        assert len(categories) >= 15

    def test_selection_provenance_is_recorded_per_benchmark(self, manifest):
        for benchmark in datasets.BENCHMARKS:
            entry = manifest["per_benchmark"][benchmark]
            assert entry["n_in_band"] >= 60
            assert entry["cohort_size"] >= 1
            assert entry["selected"] == 60


class TestQuestionConversion:
    def test_items_become_policy_facing_questions(self):
        items = datasets.load_items("mmlu_pro", "test")
        questions = list(datasets.iter_questions(items))
        assert len(questions) == TEST_PER_BENCHMARK
        assert all(question.total == TEST_PER_BENCHMARK for question in questions)
        assert [question.index for question in questions] == list(range(len(questions)))

    def test_multiple_choice_questions_render_their_options(self):
        item = next(
            item
            for item in datasets.load_items("mmlu_pro", "test")
            if item.answer_format == "mcq_single"
        )
        question = datasets.to_question(item)
        rendered = question.rendered_choices()
        assert rendered.startswith("A) ")
        assert len(rendered.splitlines()) == len(question.choices or ())

    def test_public_record_is_the_only_thing_the_host_receives(self):
        item = datasets.load_items("gsm8k", "test")[0]
        record = datasets.public_record(item)
        assert "target" not in record
        assert record["item_id"] == item.item_id
