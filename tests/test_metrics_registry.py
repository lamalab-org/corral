"""Tests for the metric registry.

This module tests:
- MetricRegistry registration and unregistration
- Metric lookup and querying
- Category-based organization
- Batch calculation of metrics
- Parallel batch calculation
- Error handling
"""

import time
from threading import Lock

import pytest

from corral.report.metrics.base import Metric, MetricMetadata, TaskMetric
from corral.report.metrics.registry import MetricRegistry, get_metrics_registry
from corral.report.results import BenchmarkResult, TaskTrialResult, TaskTrialResults


# Test Metrics
class DummyMetric(Metric):
    """A simple test metric."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="dummy_metric",
            display_name="Dummy Metric",
            description="A dummy metric for testing",
        )

    def calculate(self, benchmark_result: BenchmarkResult) -> float:
        return 42.0


class AnotherDummyMetric(Metric):
    """Another simple test metric."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="another_dummy",
            display_name="Another Dummy",
            description="Another dummy metric",
        )

    def calculate(self, benchmark_result: BenchmarkResult) -> float:
        return 10.5


class ErrorMetric(Metric):
    """A metric that raises an error during calculation."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="error_metric",
            display_name="Error Metric",
            description="A metric that errors",
        )

    def calculate(self, benchmark_result: BenchmarkResult) -> float:
        raise RuntimeError("Intentional error for testing")


class DummyTaskMetric(TaskMetric):
    """A simple task metric for testing."""

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="dummy_task_metric",
            display_name="Dummy Task Metric",
            description="A dummy task metric",
        )

    def calculate_for_task(
        self, benchmark_result: BenchmarkResult, task_id: str
    ) -> float:
        return 1.0


class SlowMetric(Metric):
    """A metric that sleeps to simulate slow calculation."""

    def __init__(self, name: str, sleep_time: float = 0.1):
        self.name_val = name
        self.sleep_time = sleep_time

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name=self.name_val,
            display_name=f"Slow Metric {self.name_val}",
            description="A slow metric for testing parallel execution",
        )

    def calculate(self, benchmark_result: BenchmarkResult) -> float:
        time.sleep(self.sleep_time)
        return len(benchmark_result.task_results) * 10.0


class CountingMetric(Metric):
    """A metric that counts how many times it's been called."""

    call_count_lock = Lock()
    call_count = 0

    def __init__(self, name: str):
        self.name_val = name

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name=self.name_val,
            display_name=f"Counting Metric {self.name_val}",
            description="A metric that counts calls",
        )

    def calculate(self, benchmark_result: BenchmarkResult) -> int:
        with CountingMetric.call_count_lock:
            CountingMetric.call_count += 1
        return 42

    @classmethod
    def reset_count(cls):
        """Reset the call counter."""
        with cls.call_count_lock:
            cls.call_count = 0


class ErrorProneMetric(Metric):
    """A metric that errors in parallel mode for testing."""

    def __init__(self, name: str):
        self.name_val = name

    @property
    def metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name=self.name_val,
            display_name=f"Error Metric {self.name_val}",
            description="A metric that raises errors",
        )

    def calculate(self, benchmark_result: BenchmarkResult) -> float:
        raise RuntimeError(f"Intentional error from {self.name_val}")


@pytest.fixture()
def empty_registry():
    """Create a fresh empty registry for testing."""
    return MetricRegistry()


@pytest.fixture()
def populated_registry():
    """Create a registry with some metrics registered."""
    registry = MetricRegistry()
    registry.register(DummyMetric())
    registry.register(AnotherDummyMetric())
    registry.register(DummyTaskMetric())
    return registry


@pytest.fixture()
def sample_benchmark_result():
    """Create a sample BenchmarkResult for testing."""
    return BenchmarkResult(
        task_results={
            "test_task": TaskTrialResults(
                task_id="test_task",
                trials=[
                    TaskTrialResult(
                        task_id="test_task",
                        trial_id="trial1",
                        score=0.8,
                        state={},
                        tool_statistics={},
                        duration=1.5,
                    )
                ],
            )
        }
    )


@pytest.fixture()
def multi_task_benchmark_result():
    """Create a BenchmarkResult with multiple tasks for parallel testing."""
    return BenchmarkResult(
        task_results={
            "task1": TaskTrialResults(
                task_id="task1",
                trials=[
                    TaskTrialResult(
                        task_id="task1",
                        trial_id="trial1",
                        score=0.8,
                        state={},
                        tool_statistics={},
                        duration=1.5,
                    )
                ],
            ),
            "task2": TaskTrialResults(
                task_id="task2",
                trials=[
                    TaskTrialResult(
                        task_id="task2",
                        trial_id="trial1",
                        score=0.9,
                        state={},
                        tool_statistics={},
                        duration=2.0,
                    )
                ],
            ),
        }
    )


class TestMetricRegistry:
    """Test MetricRegistry class."""

    def test_init_creates_empty_registry(self, empty_registry):
        """Test that a new registry is empty."""
        assert len(empty_registry.list_all()) == 0

    def test_register_metric(self, empty_registry):
        """Test registering a metric."""
        metric = DummyMetric()
        empty_registry.register(metric)

        assert len(empty_registry.list_all()) == 1
        assert empty_registry.get("dummy_metric") == metric

    def test_register_duplicate_metric_raises_error(self, empty_registry):
        """Test that registering a duplicate metric raises an error."""
        metric1 = DummyMetric()
        metric2 = DummyMetric()

        empty_registry.register(metric1)

        with pytest.raises(ValueError, match="already registered"):
            empty_registry.register(metric2)

    def test_unregister_metric(self, populated_registry):
        """Test unregistering a metric."""
        populated_registry.unregister("dummy_metric")

        with pytest.raises(KeyError, match="not found in registry"):
            populated_registry.get("dummy_metric")

        # Should still have the other metrics
        assert len(populated_registry.list_all()) == 2

    def test_unregister_nonexistent_metric_raises_error(self, empty_registry):
        """Test that unregistering a non-existent metric raises KeyError."""
        with pytest.raises(KeyError, match="not found in registry"):
            empty_registry.unregister("nonexistent")

    def test_get_metric(self, populated_registry):
        """Test getting a metric by name."""
        metric = populated_registry.get("dummy_metric")
        assert isinstance(metric, DummyMetric)
        assert metric.metadata.name == "dummy_metric"

    def test_get_nonexistent_metric_raises_error(self, empty_registry):
        """Test that getting a non-existent metric raises KeyError."""
        with pytest.raises(KeyError, match="not found in registry"):
            empty_registry.get("nonexistent")

    def test_list_all(self, populated_registry):
        """Test listing all metrics."""
        all_metrics = populated_registry.list_all()

        assert len(all_metrics) == 3
        names = {m.metadata.name for m in all_metrics}
        assert names == {"dummy_metric", "another_dummy", "dummy_task_metric"}

    def test_calculate_all(self, populated_registry, sample_benchmark_result):
        """Test calculating all metrics."""
        results = populated_registry.calculate_all(sample_benchmark_result)

        assert "dummy_metric" in results
        assert "another_dummy" in results
        assert "dummy_task_metric" in results

        assert results["dummy_metric"] == 42.0
        assert results["another_dummy"] == 10.5
        assert isinstance(results["dummy_task_metric"], dict)

    def test_calculate_all_with_enabled_only(
        self, populated_registry, sample_benchmark_result
    ):
        """Test calculating only enabled metrics."""
        results = populated_registry.calculate_all(
            sample_benchmark_result, enabled_only=["dummy_metric"]
        )

        assert "dummy_metric" in results
        assert "another_dummy" not in results
        assert "dummy_task_metric" not in results

        assert results["dummy_metric"] == 42.0

    def test_calculate_all_with_error_metric(
        self, empty_registry, sample_benchmark_result
    ):
        """Test that metrics with calculation errors are handled gracefully."""
        empty_registry.register(ErrorMetric())
        empty_registry.register(DummyMetric())

        results = empty_registry.calculate_all(sample_benchmark_result)

        assert results["error_metric"] is None  # Calculation failed
        assert results["dummy_metric"] == 42.0  # Should still succeed

    def test_calculate_all_with_error(self, empty_registry, sample_benchmark_result):
        """Test that errors during calculation are handled gracefully."""
        empty_registry.register(ErrorMetric())
        empty_registry.register(DummyMetric())

        results = empty_registry.calculate_all(sample_benchmark_result)

        assert results["error_metric"] is None  # Error was caught
        assert results["dummy_metric"] == 42.0  # Should still succeed

    def test_calculate_all_with_nonexistent_enabled_metric(
        self, populated_registry, sample_benchmark_result
    ):
        """Test that non-existent metrics in enabled_only are ignored."""
        results = populated_registry.calculate_all(
            sample_benchmark_result, enabled_only=["dummy_metric", "nonexistent"]
        )

        assert "dummy_metric" in results
        assert "nonexistent" not in results


class TestGlobalRegistry:
    """Test the global registry instance."""

    def test_get_metrics_registry_returns_same_instance(self):
        """Test that get_metrics_registry() returns the same instance."""
        registry1 = get_metrics_registry()
        registry2 = get_metrics_registry()

        assert registry1 is registry2

    def test_global_registry_is_singleton(self):
        """Test that the global registry is a singleton."""
        # Register a metric
        metric = DummyMetric()
        registry1 = get_metrics_registry()
        registry1.register(metric)

        # Get registry again and check metric is there
        registry2 = get_metrics_registry()
        assert registry2.get("dummy_metric") == metric

        # Clean up
        registry1.unregister("dummy_metric")


class TestMetricRegistryEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_enabled_only_list(self, populated_registry, sample_benchmark_result):
        """Test that an empty enabled_only list results in no calculations."""
        results = populated_registry.calculate_all(
            sample_benchmark_result, enabled_only=[]
        )
        assert len(results) == 0

    def test_none_enabled_only_calculates_all(
        self, populated_registry, sample_benchmark_result
    ):
        """Test that enabled_only=None calculates all metrics."""
        results = populated_registry.calculate_all(
            sample_benchmark_result, enabled_only=None
        )
        assert len(results) == 3


class TestParallelBatchCalculation:
    """Test parallel batch calculation of metrics."""

    def test_parallel_produces_same_results_as_sequential(
        self, multi_task_benchmark_result
    ):
        """Test that parallel mode produces identical results to sequential."""
        registry = MetricRegistry()

        # Register multiple metrics
        for i in range(5):
            metric = SlowMetric(f"metric_{i}", sleep_time=0.01)
            registry.register(metric)

        # Calculate sequentially
        sequential_results = registry.calculate_all(
            multi_task_benchmark_result, parallel=False
        )

        # Calculate in parallel
        parallel_results = registry.calculate_all(
            multi_task_benchmark_result, parallel=True
        )

        # Results should be identical
        assert sequential_results == parallel_results
        assert len(sequential_results) == 5

        # All metrics should return the same value (2 tasks * 10)
        for i in range(5):
            assert sequential_results[f"metric_{i}"] == 20.0
            assert parallel_results[f"metric_{i}"] == 20.0

    def test_parallel_is_faster_than_sequential(self, multi_task_benchmark_result):
        """Test that parallel execution is faster with multiple slow metrics."""
        registry = MetricRegistry()

        # Register multiple slow metrics
        # Using longer sleep times and more metrics to offset multiprocessing overhead
        # Multiprocessing has significant process creation overhead on macOS (spawn method)
        num_metrics = 8
        sleep_time = 1.0
        for i in range(num_metrics):
            metric = SlowMetric(f"slow_{i}", sleep_time=sleep_time)
            registry.register(metric)

        # Time sequential execution
        start_seq = time.time()
        registry.calculate_all(multi_task_benchmark_result, parallel=False)
        duration_seq = time.time() - start_seq

        # Time parallel execution
        start_par = time.time()
        registry.calculate_all(multi_task_benchmark_result, parallel=True)
        duration_par = time.time() - start_par

        # Parallel should be significantly faster
        # (allowing some overhead for process creation)
        expected_seq_time = num_metrics * sleep_time
        assert duration_seq >= expected_seq_time * 0.9  # 90% of expected

        # Parallel should be faster than sequential
        # With 8 metrics @ 1s each, sequential takes ~8s, parallel should be significantly less
        assert duration_par < duration_seq  # Just verify parallel is faster

    def test_parallel_with_enabled_only(self, multi_task_benchmark_result):
        """Test that enabled_only works correctly in parallel mode."""
        registry = MetricRegistry()

        # Register multiple metrics
        for i in range(5):
            metric = SlowMetric(f"metric_{i}", sleep_time=0.01)
            registry.register(metric)

        # Calculate only subset in parallel
        results = registry.calculate_all(
            multi_task_benchmark_result,
            enabled_only=["metric_0", "metric_2", "metric_4"],
            parallel=True,
        )

        # Should only have 3 results
        assert len(results) == 3
        assert "metric_0" in results
        assert "metric_2" in results
        assert "metric_4" in results
        assert "metric_1" not in results
        assert "metric_3" not in results

    def test_parallel_with_custom_max_workers(self, multi_task_benchmark_result):
        """Test that custom max_workers parameter is respected."""
        registry = MetricRegistry()

        # Register metrics
        for i in range(4):
            metric = SlowMetric(f"metric_{i}", sleep_time=0.05)
            registry.register(metric)

        # Calculate with max_workers=2
        results = registry.calculate_all(
            multi_task_benchmark_result, parallel=True, max_workers=2
        )

        # Should still get all results
        assert len(results) == 4
        for i in range(4):
            assert results[f"metric_{i}"] == 20.0

    def test_parallel_error_handling(self, multi_task_benchmark_result):
        """Test that errors are handled correctly in parallel mode."""
        registry = MetricRegistry()

        # Register mix of working and error-prone metrics
        registry.register(SlowMetric("good_1", sleep_time=0.01))
        registry.register(ErrorProneMetric("bad_1"))
        registry.register(SlowMetric("good_2", sleep_time=0.01))
        registry.register(ErrorProneMetric("bad_2"))

        # Calculate in parallel
        results = registry.calculate_all(multi_task_benchmark_result, parallel=True)

        # Good metrics should succeed
        assert results["good_1"] == 20.0
        assert results["good_2"] == 20.0

        # Error metrics should return None
        assert results["bad_1"] is None
        assert results["bad_2"] is None

    def test_parallel_with_empty_metrics_list(self, multi_task_benchmark_result):
        """Test parallel calculation with no metrics."""
        registry = MetricRegistry()

        results = registry.calculate_all(multi_task_benchmark_result, parallel=True)

        assert results == {}

    def test_parallel_with_single_metric(self, multi_task_benchmark_result):
        """Test that parallel works correctly with just one metric."""
        registry = MetricRegistry()
        registry.register(SlowMetric("single", sleep_time=0.01))

        results = registry.calculate_all(multi_task_benchmark_result, parallel=True)

        assert len(results) == 1
        assert results["single"] == 20.0

    def test_process_safety(self, multi_task_benchmark_result):
        """Test that parallel execution with multiprocessing works correctly."""
        registry = MetricRegistry()

        # Register multiple counting metrics
        num_metrics = 10
        for i in range(num_metrics):
            registry.register(CountingMetric(f"counter_{i}"))

        # Calculate in parallel
        results = registry.calculate_all(multi_task_benchmark_result, parallel=True)

        # All metrics should have returned results
        # Note: With multiprocessing, class variables are not shared across processes,
        # so we verify results instead of call counts
        assert len(results) == num_metrics

        # All should return the same value (42 is what CountingMetric returns)
        for i in range(num_metrics):
            assert results[f"counter_{i}"] == 42

    def test_sequential_still_works(self, multi_task_benchmark_result):
        """Test that sequential mode (default) still works correctly."""
        registry = MetricRegistry()

        # Register metrics
        for i in range(3):
            metric = SlowMetric(f"metric_{i}", sleep_time=0.01)
            registry.register(metric)

        # Calculate without parallel flag (default sequential)
        results = registry.calculate_all(multi_task_benchmark_result)

        assert len(results) == 3
        for i in range(3):
            assert results[f"metric_{i}"] == 20.0

    def test_parallel_false_explicitly(self, multi_task_benchmark_result):
        """Test that parallel=False explicitly uses sequential calculation."""
        registry = MetricRegistry()

        # Register metrics
        for i in range(3):
            metric = SlowMetric(f"metric_{i}", sleep_time=0.01)
            registry.register(metric)

        # Explicitly set parallel=False
        results = registry.calculate_all(multi_task_benchmark_result, parallel=False)

        assert len(results) == 3
        for i in range(3):
            assert results[f"metric_{i}"] == 20.0


class TestBatchCalculationBackwardCompatibility:
    """Test that existing usage patterns still work."""

    def test_backward_compatible_call_signature(self, multi_task_benchmark_result):
        """Test that old call signature (without parallel arg) still works."""
        registry = MetricRegistry()
        registry.register(SlowMetric("test", sleep_time=0.01))

        # Old style call - should work fine
        results = registry.calculate_all(multi_task_benchmark_result)
        assert results["test"] == 20.0

        # With enabled_only
        results = registry.calculate_all(
            multi_task_benchmark_result, enabled_only=["test"]
        )
        assert results["test"] == 20.0
