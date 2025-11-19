# Roadmap: Metrics Registry Refactoring

## Overview

This roadmap outlines the steps needed to complete the transition to a registry-based metrics system. The goal is to make metrics fully modular, allowing users to register custom metrics or unregister default ones via a clean API or entry points.

## Current State

✅ **Completed:**

- `MetricRegistry` class implemented with registration/unregistration
- `Metric` and `TaskMetric` base classes with `MetricMetadata`
- Core metrics implemented in `core.py` (independent of `BenchmarkResult`)
- Default metrics auto-registered on import via `register_default_metrics()`
- Plugin discovery via entry points implemented (`discover_plugin_metrics()`)
- Registry exposed via `get_registry()` function
- `calculate_all()` method with parallel execution support
- Comprehensive test coverage for registry functionality

⚠️ **In Progress:**

- Metrics exist in both `BenchmarkResult` methods (deprecated) and `core.py`
- User-facing API not yet exposed in `__init__.py`
- Report generation uses registry but also has fallback logic

## Remaining Work

### Phase 1: Complete Deprecation of BenchmarkResult Methods

**Goal:** Fully remove metric calculation logic from `BenchmarkResult` class methods.

#### Tasks

1. **Update `BenchmarkResult` methods to pure wrappers** (`src/corral/report/results.py`)
   - Keep method signatures for backward compatibility
   - Replace implementation with direct registry calls
   - Example:

     ```python
     def average_score(self) -> float:
         """Calculate average score across all results

         .. deprecated:: 0.2.0
             Use the metrics registry instead.
         """
         warnings.warn(...)
         metric = self.metric_registry.get("average_score")
         return metric.calculate(self)
     ```

   - Apply to all deprecated methods:
     - `average_score()`
     - `overall_success_rate()`
     - `total_surrendered_trials()`
     - `overall_total_duration()`
     - `overall_average_duration()`
     - `total_tool_execution_duration()`
     - `total_token_usage()`
     - `total_tool_calls()`
     - `task_success_rate()`
     - `task_average_duration()`
     - `task_total_token_usage()`
     - `task_pass_at_k()`
     - `task_pass_hat_k()`

2. **Helper methods cleanup**
   - Keep only essential helper methods in `BenchmarkResult`:
     - `_sum_token_usage()` - if still needed by other code
     - `get_trial_tool_execution_duration()` - used in report generation
     - `_calculate_task_average_score()` - used in report generation
   - Move any reusable logic to `core.py` if it can be generalized

3. **Update tests**
   - Ensure existing tests still pass with wrapper methods
   - Add tests verifying deprecation warnings are raised
   - Update test assertions to check registry calls

### Phase 2: Expose User-Facing API

**Goal:** Make the metrics registry accessible and discoverable for end users.

#### Tasks

1. **Update `src/corral/__init__.py`**
   - Export registry access functions:

     ```python
     from .report.metrics import (
         get_registry,
         register_default_metrics,
         discover_plugin_metrics,
         Metric,
         TaskMetric,
         MetricMetadata,
     )

     __all__ = [
         "CorralRouter",
         "CorralRunner",
         # Metrics API
         "get_registry",
         "register_default_metrics",
         "discover_plugin_metrics",
         "Metric",
         "TaskMetric",
         "MetricMetadata",
     ]
     ```

2. **Create user-facing examples**
   - Add example in `docs/` showing:
     - How to register a custom metric
     - How to unregister a default metric
     - How to list available metrics
     - How to enable/disable specific metrics
   - Example code:

     ```python
     from corral import get_registry
     from corral.report.metrics import Metric, MetricMetadata


     # Define custom metric
     class MyCustomMetric(Metric):
         @property
         def metadata(self):
             return MetricMetadata(
                 name="my_metric",
                 display_name="My Custom Metric",
                 description="A custom metric",
             )

         def calculate(self, benchmark_result):
             return 42.0


     # Register it
     registry = get_registry()
     registry.register(MyCustomMetric())

     # Unregister default metric
     registry.unregister("average_score")
     ```

### Phase 3: Plugin System Documentation

Deprecated

### Phase 4: Enhanced Report Generation

**Goal:** Make report generation fully registry-driven with better customization.

#### Tasks

1. **Update `_prepare_report_data()`**
   - Remove hardcoded metric calculations
   - Use only `calculate_metrics()` results
   - Ensure all task-level metrics use registry
   - Clean up legacy fallback code

### Phase 5: Testing and Validation

**Goal:** Ensure robustness and backward compatibility.

#### Tasks

1. **Comprehensive integration tests**
   - Test full benchmark runs with custom metrics
   - Test entry point discovery with actual packages
   - Test enable/disable functionality end-to-end
   - Test report generation with various metric combinations

2. **Performance testing**
   - Benchmark parallel vs sequential calculation
   - Ensure registry overhead is minimal
   - Test with large numbers of metrics

3. **Backward compatibility tests**
   - Verify all deprecated methods still work
   - Check that existing user code doesn't break
   - Test migration path for users

### Phase 6: Documentation and Examples

**Goal:** Complete user-facing documentation.

#### Tasks

1. **API documentation**
   - Document all public registry methods
   - Add docstring examples for each method
   - Document the `Metric` and `TaskMetric` protocols

2. **Tutorial/Guide**
   - "Getting Started with Custom Metrics"
   - "Migrating from Direct Methods to Registry"
   - "Best Practices for Metric Development"

3. **Update README**
   - Add section on metrics system
   - Link to detailed documentation
   - Show quick example of custom metric

## Migration Timeline

### Short Term (Current Sprint)

- ✅ Phase 1: Complete method deprecation
- ✅ Phase 2: Expose user API

### Medium Term (Next Sprint)

- Phase 3: Plugin documentation
- Phase 4: Enhanced reporting

### Long Term (Future)

- Phase 5: Comprehensive testing
- Phase 6: Full documentation
- Consider removing deprecated methods in v1.0

## Technical Considerations

### Backward Compatibility

- All existing `BenchmarkResult` methods must continue working
- Deprecation warnings guide users to new API
- Consider multi-version deprecation strategy

### Performance

- Registry lookup is fast (dict-based)
- Parallel calculation available for heavy workloads
- Lazy calculation - only compute requested metrics

### Extensibility

- Entry points allow third-party plugins
- `Metric` and `TaskMetric` protocols are easy to implement
- No dependency on internal Corral structures

## Success Metrics

The refactoring will be considered complete when:

1. ✅ All default metrics can be calculated via registry
2. ⬜ Users can register custom metrics without modifying Corral code
3. ⬜ Users can install metric plugins via pip
4. ⬜ All deprecated methods are thin wrappers around registry
5. ⬜ Documentation covers custom metric development
6. ⬜ Report generation is fully registry-driven

## Open Questions

1. **Metric Categories**: Should we expand the category system for better organization?
2. **Configuration Format**: What's the best way to configure enabled metrics (CLI flags, config file, programmatic)?
3. **Metric Dependencies**: How to handle metrics that depend on other metrics?
4. **Caching**: Should metric results be cached during a benchmark run?
5. **Versioning**: How to handle metric API changes in plugins?

## Notes

- The registry pattern provides excellent separation of concerns
- Entry points enable ecosystem growth without core changes
- Current implementation is solid, mainly needs exposure and documentation
- Parallel calculation is a nice-to-have for heavy workloads
