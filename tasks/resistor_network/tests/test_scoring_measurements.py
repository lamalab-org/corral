import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.strategies import composite

from resistor_network.score import _simulate_resistance, check_resistance_measurements


class TestSimulateResistance:
    """Unit tests for _simulate_resistance function"""

    def test_single_resistor(self):
        """Test circuit with single resistor between two nodes"""
        topology = {"resistors": {"R1": 100.0}, "connections": [["A", "B", "R1"]]}

        resistance = _simulate_resistance(topology, "A", "B")
        assert abs(resistance - 100.0) < 1e-6

    def test_series_resistors(self):
        """Test series circuit: A---R1---B---R2---C"""
        topology = {
            "resistors": {"R1": 10.0, "R2": 20.0},
            "connections": [["A", "B", "R1"], ["B", "C", "R2"]],
        }

        # Test individual segments
        assert abs(_simulate_resistance(topology, "A", "B") - 10.0) < 1e-6
        assert abs(_simulate_resistance(topology, "B", "C") - 20.0) < 1e-6
        # Test total series resistance
        assert abs(_simulate_resistance(topology, "A", "C") - 30.0) < 1e-6

    def test_parallel_resistors(self):
        """Test parallel circuit: A connected to B via two parallel paths"""
        topology = {
            "resistors": {"R1": 10.0, "R2": 10.0},
            "connections": [["A", "B", "R1"], ["A", "B", "R2"]],
        }

        # Parallel resistance: 1/R_total = 1/R1 + 1/R2
        # For R1=R2=10Ω: R_total = 5Ω
        resistance = _simulate_resistance(topology, "A", "B")
        assert abs(resistance - 5.0) < 1e-6

    def test_complex_circuit(self):
        """Test more complex circuit topology"""
        topology = {
            "resistors": {"R1": 10.0, "R2": 20.0, "R3": 30.0},
            "connections": [
                ["A", "B", "R1"],  # 10Ω
                ["A", "C", "R2"],  # 20Ω
                ["B", "C", "R3"],  # 30Ω
            ],
        }

        # This creates a triangle of resistors
        resistance_ab = _simulate_resistance(topology, "A", "B")
        resistance_ac = _simulate_resistance(topology, "A", "C")
        resistance_bc = _simulate_resistance(topology, "B", "C")

        # All resistances should be positive and finite
        assert resistance_ab > 0
        assert math.isfinite(resistance_ab)
        assert resistance_ac > 0
        assert math.isfinite(resistance_ac)
        assert resistance_bc > 0
        assert math.isfinite(resistance_bc)

        # Resistance A-B should be less than R1 (10Ω) due to parallel path via C
        assert resistance_ab < 10.0

    def test_same_node_resistance(self):
        """Test resistance between same node (should be 0)"""
        topology = {"resistors": {"R1": 100.0}, "connections": [["A", "B", "R1"]]}

        resistance = _simulate_resistance(topology, "A", "A")
        assert abs(resistance) < 1e-6

    def test_symmetry(self):
        """Test that resistance A-B equals resistance B-A"""
        topology = {
            "resistors": {"R1": 15.0, "R2": 25.0},
            "connections": [["A", "B", "R1"], ["B", "C", "R2"]],
        }

        resistance_ab = _simulate_resistance(topology, "A", "B")
        resistance_ba = _simulate_resistance(topology, "B", "A")

        assert abs(resistance_ab - resistance_ba) < 1e-6

    def test_invalid_nodes(self):
        """Test behavior with non-existent nodes"""
        topology = {"resistors": {"R1": 100.0}, "connections": [["A", "B", "R1"]]}

        # This should raise an exception or return inf
        result = _simulate_resistance(topology, "X", "Y")
        assert math.isinf(result) or result is None

    def test_empty_topology(self):
        """Test behavior with empty topology"""
        topology = {"resistors": {}, "connections": []}

        result = _simulate_resistance(topology, "A", "B")
        assert math.isinf(result)


class TestCheckResistanceMeasurements:
    """Unit tests for check_resistance_measurements function"""

    def test_returns_callable(self):
        """Test that function returns a callable"""
        measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]
        scorer = check_resistance_measurements(measurements)
        assert callable(scorer)

    def test_perfect_match_single_measurement(self):
        """Test perfect match for single measurement"""
        measurements = [{"node_a": "A", "node_b": "B", "resistance": 100.0}]
        scorer = check_resistance_measurements(measurements, tolerance=0.05)

        topology = {"resistors": {"R1": 100.0}, "connections": [["A", "B", "R1"]]}

        score = scorer(json.dumps(topology))
        assert abs(score - 1.0) < 1e-6

    def test_perfect_match_multiple_measurements(self):
        """Test perfect match for multiple measurements"""
        measurements = [
            {"node_a": "A", "node_b": "B", "resistance": 10.0},
            {"node_a": "B", "node_b": "C", "resistance": 20.0},
            {"node_a": "A", "node_b": "C", "resistance": 30.0},
        ]
        scorer = check_resistance_measurements(measurements, tolerance=0.05)

        topology = {
            "resistors": {"R1": 10.0, "R2": 20.0},
            "connections": [["A", "B", "R1"], ["B", "C", "R2"]],
        }

        score = scorer(json.dumps(topology))
        assert abs(score - 1.0) < 1e-6

    def test_within_tolerance(self):
        """Test scoring within tolerance"""
        measurements = [{"node_a": "A", "node_b": "B", "resistance": 100.0}]
        scorer = check_resistance_measurements(
            measurements, tolerance=0.1
        )  # 10% tolerance

        # Create topology with 5% error (should get high score)
        topology = {
            "resistors": {"R1": 105.0},  # 5% higher than expected
            "connections": [["A", "B", "R1"]],
        }

        score = scorer(json.dumps(topology))
        assert score > 0.5  # Should get decent score within tolerance

    def test_outside_tolerance(self):
        """Test scoring outside tolerance"""
        measurements = [{"node_a": "A", "node_b": "B", "resistance": 100.0}]
        scorer = check_resistance_measurements(
            measurements, tolerance=0.05
        )  # 5% tolerance

        # Create topology with 20% error (should get low score)
        topology = {
            "resistors": {"R1": 120.0},  # 20% higher than expected
            "connections": [["A", "B", "R1"]],
        }

        score = scorer(json.dumps(topology))
        assert score < 0.5  # Should get low score outside tolerance

    def test_zero_resistance_measurement(self):
        """Test handling of zero resistance measurement"""
        measurements = [{"node_a": "A", "node_b": "B", "resistance": 0.0}]
        scorer = check_resistance_measurements(measurements)

        # Create topology that should give near-zero resistance
        topology = {
            "resistors": {"R1": 1e-9},  # Very small resistance
            "connections": [["A", "B", "R1"]],
        }

        score = scorer(json.dumps(topology))
        assert score > 0.9  # Should get high score for near-zero

    def test_file_input(self):
        """Test scoring with file input"""
        measurements = [{"node_a": "A", "node_b": "B", "resistance": 50.0}]
        scorer = check_resistance_measurements(measurements)

        topology = {"resistors": {"R1": 50.0}, "connections": [["A", "B", "R1"]]}

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(topology, f)
            temp_path = f.name

        try:
            score = scorer(temp_path)
            assert abs(score - 1.0) < 1e-6
        finally:
            Path(temp_path).unlink()  # Clean up

    def test_invalid_json_input(self):
        """Test handling of invalid JSON input"""
        measurements = [{"node_a": "A", "node_b": "B", "resistance": 50.0}]
        scorer = check_resistance_measurements(measurements)

        score = scorer("invalid json {")
        assert score == 0.0

    def test_empty_measurements(self):
        """Test behavior with empty measurements list"""
        scorer = check_resistance_measurements([])

        topology = {"resistors": {"R1": 50.0}, "connections": [["A", "B", "R1"]]}

        # Should return 0.0 if no measurements to check against
        score = scorer(json.dumps(topology))
        assert score == 0.0

    @patch("resistor_network.score._simulate_resistance")
    def test_simulation_exception_handling(self, mock_simulate):
        """Test handling of simulation exceptions"""
        mock_simulate.side_effect = Exception("Simulation failed")

        measurements = [{"node_a": "A", "node_b": "B", "resistance": 50.0}]
        scorer = check_resistance_measurements(measurements)

        topology = {"resistors": {"R1": 50.0}, "connections": [["A", "B", "R1"]]}

        score = scorer(json.dumps(topology))
        assert score == 0.0  # Should return 0 when simulation fails


# Hypothesis-based property tests
@composite
def topology_with_single_resistor(draw):
    """Generate a simple topology with one resistor"""
    resistance = draw(
        st.floats(
            min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False
        )
    )
    node_a = draw(st.text(min_size=1, max_size=3, alphabet="ABCDEFGH"))
    node_b = draw(st.text(min_size=1, max_size=3, alphabet="ABCDEFGH"))
    assume(node_a != node_b)

    topology = {
        "resistors": {"R1": resistance},
        "connections": [[node_a, node_b, "R1"]],
    }
    return topology, node_a, node_b, resistance


@composite
def series_topology(draw):
    """Generate a series circuit topology"""
    r1 = draw(
        st.floats(
            min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False
        )
    )
    r2 = draw(
        st.floats(
            min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False
        )
    )

    topology = {
        "resistors": {"R1": r1, "R2": r2},
        "connections": [["A", "B", "R1"], ["B", "C", "R2"]],
    }
    return topology, r1, r2


class TestHypothesisProperties:
    """Property-based tests using Hypothesis"""

    @given(topology_with_single_resistor())
    def test_single_resistor_property(self, topology_data):
        """Property: Single resistor should give expected resistance"""
        topology, node_a, node_b, expected_resistance = topology_data

        result = _simulate_resistance(topology, node_a, node_b)
        assert abs(result - expected_resistance) < 1e-6

    @given(topology_with_single_resistor())
    def test_resistance_symmetry_property(self, topology_data):
        """Property: Resistance A-B should equal resistance B-A"""
        topology, node_a, node_b, _ = topology_data

        resistance_ab = _simulate_resistance(topology, node_a, node_b)
        resistance_ba = _simulate_resistance(topology, node_b, node_a)

        assert abs(resistance_ab - resistance_ba) < 1e-6

    @given(topology_with_single_resistor())
    def test_self_resistance_property(self, topology_data):
        """Property: Resistance from node to itself should be zero"""
        topology, node_a, _, _ = topology_data

        resistance = _simulate_resistance(topology, node_a, node_a)
        assert abs(resistance) < 1e-6

    @given(series_topology())
    def test_series_resistance_property(self, topology_data):
        """Property: Series resistance should equal sum of individual resistances"""
        topology, r1, r2 = topology_data

        total_resistance = _simulate_resistance(topology, "A", "C")
        expected_total = r1 + r2

        assert abs(total_resistance - expected_total) < 1e-6

    @given(
        st.lists(
            st.fixed_dictionaries(
                {
                    "node_a": st.text(min_size=1, max_size=2, alphabet="AB"),
                    "node_b": st.text(min_size=1, max_size=2, alphabet="AB"),
                    "resistance": st.floats(
                        min_value=0.1,
                        max_value=100.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                }
            ),
            min_size=1,
            max_size=5,
        ),
        st.floats(min_value=0.01, max_value=0.5),
    )
    def test_scorer_bounds_property(self, measurements, tolerance):
        """Property: Scorer should always return values between 0.0 and 1.0"""
        # Filter out measurements where node_a == node_b
        valid_measurements = [m for m in measurements if m["node_a"] != m["node_b"]]
        assume(len(valid_measurements) > 0)

        scorer = check_resistance_measurements(valid_measurements, tolerance)

        # Test with a simple topology
        topology = {"resistors": {"R1": 50.0}, "connections": [["A", "B", "R1"]]}

        score = scorer(json.dumps(topology))
        assert 0.0 <= score <= 1.0

    @given(topology_with_single_resistor(), st.floats(min_value=0.01, max_value=0.2))
    def test_perfect_score_property(self, topology_data, tolerance):
        """Property: Perfect match should give score of 1.0"""
        topology, node_a, node_b, resistance = topology_data

        measurements = [{"node_a": node_a, "node_b": node_b, "resistance": resistance}]
        scorer = check_resistance_measurements(measurements, tolerance)

        score = scorer(json.dumps(topology))
        assert abs(score - 1.0) < 1e-6
