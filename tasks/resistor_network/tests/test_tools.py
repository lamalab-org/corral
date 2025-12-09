import json
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st
from resistor_network.tools import (
    delta_to_wye_transform,
    estimate_resistor_values,
    generate_test_measurements,
    propose_simple_topology,
    simulate_circuit_resistance,
    validate_measurements,
    wye_to_delta_transform,
)


class TestDeltaToWyeTransform:
    """Tests for delta_to_wye_transform tool"""

    def test_equal_resistors_triangle(self):
        """Test delta-to-wye with equal resistors"""
        result_json = delta_to_wye_transform.execute(ra=30.0, rb=30.0, rc=30.0)
        result = json.loads(result_json)

        assert abs(result["r1"] - 10.0) < 1e-6
        assert abs(result["r2"] - 10.0) < 1e-6
        assert abs(result["r3"] - 10.0) < 1e-6

    def test_different_resistors(self):
        """Test with different resistor values"""
        result_json = delta_to_wye_transform.execute(ra=6.0, rb=3.0, rc=2.0)
        result = json.loads(result_json)

        # Calculate expected values: r1 = (rb*rc)/(ra+rb+rc)
        total = 6.0 + 3.0 + 2.0  # 11
        expected_r1 = (3.0 * 2.0) / total  # 6/11
        expected_r2 = (6.0 * 2.0) / total  # 12/11
        expected_r3 = (6.0 * 3.0) / total  # 18/11

        assert abs(result["r1"] - expected_r1) < 1e-6
        assert abs(result["r2"] - expected_r2) < 1e-6
        assert abs(result["r3"] - expected_r3) < 1e-6

    def test_zero_sum_error(self):
        """Test error when sum of resistances is zero"""
        with pytest.raises(ValueError, match="Sum of delta resistances cannot be zero"):
            delta_to_wye_transform.execute(ra=0.0, rb=0.0, rc=0.0)

    def test_negative_resistors_raises_error(self):
        """Test with negative resistors should raise ValueError"""
        with pytest.raises(ValueError, match="All resistances must be positive"):
            delta_to_wye_transform.execute(ra=10.0, rb=-5.0, rc=15.0)

    def test_large_values(self):
        """Test with large resistance values"""
        result_json = delta_to_wye_transform.execute(ra=1e6, rb=2e6, rc=3e6)
        result = json.loads(result_json)

        total = 6e6
        expected_r1 = (2e6 * 3e6) / total  # 1e6
        expected_r2 = (1e6 * 3e6) / total  # 0.5e6
        expected_r3 = (1e6 * 2e6) / total  # 0.333e6

        assert abs(result["r1"] - expected_r1) < 1e-3
        assert abs(result["r2"] - expected_r2) < 1e-3
        assert abs(result["r3"] - expected_r3) < 1e-3


class TestWyeToDeltaTransform:
    """Tests for wye_to_delta_transform tool"""

    def test_equal_resistors_star(self):
        """Test wye-to-delta with equal resistors"""
        result_json = wye_to_delta_transform.execute(r1=10.0, r2=10.0, r3=10.0)
        result = json.loads(result_json)

        # Should get 30 ohm resistors in delta
        assert abs(result["ra"] - 30.0) < 1e-6
        assert abs(result["rb"] - 30.0) < 1e-6
        assert abs(result["rc"] - 30.0) < 1e-6

    def test_different_resistors(self):
        """Test with different resistor values"""
        result_json = wye_to_delta_transform.execute(r1=2.0, r2=3.0, r3=6.0)
        result = json.loads(result_json)

        # Calculate expected: ra = (r1*r2 + r2*r3 + r3*r1) / r3
        denominator = 2.0 * 3.0 + 3.0 * 6.0 + 6.0 * 2.0  # 6 + 18 + 12 = 36
        expected_ra = denominator / 6.0  # 6
        expected_rb = denominator / 2.0  # 18
        expected_rc = denominator / 3.0  # 12

        assert abs(result["ra"] - expected_ra) < 1e-6
        assert abs(result["rb"] - expected_rb) < 1e-6
        assert abs(result["rc"] - expected_rc) < 1e-6

    def test_negative_resistor_error(self):
        """Test error with negative resistors"""
        with pytest.raises(ValueError, match="All resistances must be positive"):
            wye_to_delta_transform.execute(r1=-5.0, r2=10.0, r3=15.0)

    def test_zero_resistor_error(self):
        """Test error with zero resistor"""
        with pytest.raises(ValueError, match="All resistances must be positive"):
            wye_to_delta_transform.execute(r1=0.0, r2=10.0, r3=15.0)

    def test_roundtrip_transformation(self):
        """Test that delta->wye->delta gives original values"""
        # Start with delta
        ra_orig, rb_orig, rc_orig = 30.0, 30.0, 30.0

        # Convert to wye
        wye_result_json = delta_to_wye_transform.execute(
            ra=ra_orig, rb=rb_orig, rc=rc_orig
        )
        wye_result = json.loads(wye_result_json)

        # Convert back to delta
        delta_result_json = wye_to_delta_transform.execute(
            r1=wye_result["r1"], r2=wye_result["r2"], r3=wye_result["r3"]
        )
        delta_result = json.loads(delta_result_json)

        assert abs(delta_result["ra"] - ra_orig) < 1e-6
        assert abs(delta_result["rb"] - rb_orig) < 1e-6
        assert abs(delta_result["rc"] - rc_orig) < 1e-6


class TestSimulateCircuitResistance:
    """Tests for simulate_circuit_resistance tool"""

    def test_simple_series_circuit(self):
        """Test series circuit: A-R1-B-R2-C"""
        topology = {
            "resistors": {"R1": 10.0, "R2": 20.0},
            "connections": [["A", "B", "R1"], ["B", "C", "R2"]],
        }

        resistance = simulate_circuit_resistance.execute(
            topology=json.dumps(topology), terminal_nodes=["A", "C"]
        )
        resistance = float(resistance)

        assert abs(resistance - 30.0) < 1e-6

    def test_simple_parallel_circuit(self):
        """Test parallel circuit: A connected to B via two paths"""
        topology = {
            "resistors": {"R1": 10.0, "R2": 10.0},
            "connections": [["A", "B", "R1"], ["A", "B", "R2"]],
        }

        resistance = simulate_circuit_resistance.execute(
            topology=json.dumps(topology), terminal_nodes=["A", "B"]
        )

        # Parallel resistance: 1/R = 1/R1 + 1/R2 = 1/10 + 1/10 = 2/10, so R = 5
        assert abs(float(resistance) - 5.0) < 1e-6

    def test_series_parallel_combination(self):
        """Test series-parallel combination"""
        topology = {
            "resistors": {"R1": 10.0, "R2": 20.0, "R3": 20.0},
            "connections": [
                ["A", "B", "R1"],  # R1 in series
                ["B", "C", "R2"],  # R2 and R3 in parallel
                ["B", "C", "R3"],
            ],
        }

        resistance = simulate_circuit_resistance.execute(
            topology=json.dumps(topology), terminal_nodes=["A", "C"]
        )

        # R2||R3 = 20||20 = 10, Total = R1 + (R2||R3) = 10 + 10 = 20
        assert abs(float(resistance) - 20.0) < 1e-6

    def test_same_node_resistance(self):
        """Test resistance between same node (should be 0)"""
        topology = {"resistors": {"R1": 100.0}, "connections": [["A", "B", "R1"]]}

        resistance = simulate_circuit_resistance.execute(
            topology=json.dumps(topology), terminal_nodes=["A", "A"]
        )

        assert abs(float(resistance)) < 1e-6

    def test_invalid_json(self):
        """Test with invalid JSON input"""
        with pytest.raises(ValueError, match="Error simulating circuit"):
            simulate_circuit_resistance.execute(
                topology="invalid json", terminal_nodes=["A", "B"]
            )

    def test_missing_resistor_error(self):
        """Test error when resistor referenced in connections doesn't exist"""
        topology = {
            "resistors": {"R1": 10.0},
            "connections": [["A", "B", "R2"]],  # R2 doesn't exist
        }

        with pytest.raises(ValueError, match="Resistor R2 not found"):
            simulate_circuit_resistance.execute(
                topology=json.dumps(topology), terminal_nodes=["A", "B"]
            )

    def test_negative_resistance_error(self):
        """Test error with negative resistance"""
        topology = {"resistors": {"R1": -10.0}, "connections": [["A", "B", "R1"]]}

        with pytest.raises(ValueError, match="Resistance must be positive"):
            simulate_circuit_resistance.execute(
                topology=json.dumps(topology), terminal_nodes=["A", "B"]
            )

    def test_wrong_terminal_count(self):
        """Test error with wrong number of terminal nodes"""
        topology = {"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}

        with pytest.raises(ValueError, match="Must specify exactly two terminal nodes"):
            simulate_circuit_resistance.execute(
                topology=json.dumps(topology), terminal_nodes=["A", "B", "C"]
            )

    def test_disconnected_circuit(self):
        """Test error with disconnected circuit"""
        topology = {
            "resistors": {"R1": 10.0, "R2": 20.0},
            "connections": [
                ["A", "B", "R1"],
                ["C", "D", "R2"],
            ],  # Two separate components
        }

        with pytest.raises(ValueError, match="Circuit is not solvable"):
            simulate_circuit_resistance.execute(
                topology=json.dumps(topology), terminal_nodes=["A", "C"]
            )


class TestValidateMeasurements:
    """Tests for validate_measurements tool"""

    def test_perfect_match(self):
        """Test validation with perfect match"""
        topology = {"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}
        measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

        with patch(
            "resistor_network.tools.simulate_circuit_resistance.execute"
        ) as mock_sim:
            mock_sim.return_value = 10.0

            result_json = validate_measurements.execute(
                topology=json.dumps(topology), measurements=json.dumps(measurements)
            )
            result = json.loads(result_json)

        assert result["total_error"] == 0.0
        assert result["max_error"] == 0.0
        assert result["mean_error"] == 0.0
        assert len(result["detailed_errors"]) == 1

    def test_measurement_error(self):
        """Test validation with measurement errors"""
        topology = {"resistors": {"R1": 15.0}, "connections": [["A", "B", "R1"]]}
        measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

        with patch(
            "resistor_network.tools.simulate_circuit_resistance.execute"
        ) as mock_sim:
            mock_sim.return_value = 15.0  # Predicted differs from actual

            result_json = validate_measurements.execute(
                topology=json.dumps(topology), measurements=json.dumps(measurements)
            )
            result = json.loads(result_json)

        assert result["total_error"] == 5.0  # |15 - 10|
        assert result["max_error"] == 5.0
        assert result["mean_error"] == 5.0
        assert result["detailed_errors"][0]["relative_error"] == 0.5  # 5/10

    def test_multiple_measurements(self):
        """Test validation with multiple measurements"""
        topology = {
            "resistors": {"R1": 10.0, "R2": 20.0},
            "connections": [["A", "B", "R1"], ["B", "C", "R2"]],
        }
        measurements = [
            {"node_a": "A", "node_b": "B", "resistance": 10.0},
            {"node_a": "B", "node_b": "C", "resistance": 20.0},
            {"node_a": "A", "node_b": "C", "resistance": 30.0},
        ]

        with patch(
            "resistor_network.tools.simulate_circuit_resistance.execute"
        ) as mock_sim:
            mock_sim.side_effect = [10.0, 20.0, 30.0]  # Perfect matches

            result_json = validate_measurements.execute(
                topology=json.dumps(topology), measurements=json.dumps(measurements)
            )
            result = json.loads(result_json)

        assert result["total_error"] < abs(1e-6)
        assert result["num_measurements"] == 3
        assert len(result["detailed_errors"]) == 3

    def test_simulation_failure(self):
        """Test handling of simulation failures"""
        topology = {"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}
        measurements = [{"node_a": "X", "node_b": "Y", "resistance": 10.0}]

        with patch(
            "resistor_network.tools.simulate_circuit_resistance.execute"
        ) as mock_sim:
            mock_sim.side_effect = Exception("Node not found")

            result_json = validate_measurements.execute(
                topology=json.dumps(topology), measurements=json.dumps(measurements)
            )
            result = json.loads(result_json)

        assert result["total_error"] == 1000.0  # Large error for failed simulation
        assert "simulation_failed" in result["detailed_errors"][0]["error_type"]

    def test_invalid_json_measurements(self):
        """Test with invalid JSON measurements"""
        topology = {"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}

        result_json = validate_measurements.execute(
            topology=json.dumps(topology), measurements="invalid json"
        )
        result = json.loads(result_json)

        assert "error" in result
        assert "Validation failed" in result["error"]

    def test_zero_resistance_division(self):
        """Test relative error calculation with zero actual resistance"""
        topology = {"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}
        measurements = [{"node_a": "A", "node_b": "B", "resistance": 0.0}]

        with patch(
            "resistor_network.tools.simulate_circuit_resistance.execute"
        ) as mock_sim:
            mock_sim.return_value = 5.0

            result_json = validate_measurements.execute(
                topology=json.dumps(topology), measurements=json.dumps(measurements)
            )
            result = json.loads(result_json)

        assert result["detailed_errors"][0]["relative_error"] == float("inf")


class TestProposeSimpleTopology:
    """Tests for propose_simple_topology tool"""

    def test_series_topology(self):
        """Test generation of series topology"""
        result_json = propose_simple_topology.execute(
            num_resistors=3, topology_type="series"
        )
        result = json.loads(result_json)

        assert len(result["resistors"]) == 3
        assert "R1" in result["resistors"]
        assert "R2" in result["resistors"]
        assert "R3" in result["resistors"]

        # Check series connections: A-R1-B-R2-C-R3-D
        expected_connections = [["A", "B", "R1"], ["B", "C", "R2"], ["C", "D", "R3"]]
        assert result["connections"] == expected_connections

    def test_parallel_topology(self):
        """Test generation of parallel topology"""
        result_json = propose_simple_topology.execute(
            num_resistors=2, topology_type="parallel"
        )
        result = json.loads(result_json)

        assert len(result["resistors"]) == 2

        # All resistors should be between same nodes A and B
        for connection in result["connections"]:
            assert connection[0] == "A"
            assert connection[1] == "B"

    def test_series_parallel_topology(self):
        """Test generation of series-parallel topology"""
        result_json = propose_simple_topology.execute(
            num_resistors=3, topology_type="series_parallel"
        )
        result = json.loads(result_json)

        assert len(result["resistors"]) == 3

        # Should have R1 in series, then R2||R3 in parallel
        connections = result["connections"]
        assert ["A", "B", "R1"] in connections
        assert ["B", "C", "R2"] in connections
        assert ["B", "C", "R3"] in connections

    def test_bridge_topology(self):
        """Test generation of bridge topology"""
        result_json = propose_simple_topology.execute(
            num_resistors=5, topology_type="bridge"
        )
        result = json.loads(result_json)

        assert len(result["resistors"]) == 5

        # Check for Wheatstone bridge structure
        expected_connections = [
            ["A", "B", "R1"],
            ["A", "C", "R2"],
            ["B", "D", "R3"],
            ["C", "D", "R4"],
            ["B", "C", "R5"],  # Bridge resistor
        ]
        assert result["connections"] == expected_connections

    def test_insufficient_resistors_for_bridge(self):
        """Test bridge with insufficient resistors defaults to series"""
        result_json = propose_simple_topology.execute(
            num_resistors=3, topology_type="bridge"
        )
        result = json.loads(result_json)

        # Should fall back to series configuration
        expected_connections = [["A", "B", "R1"], ["B", "C", "R2"], ["C", "D", "R3"]]
        assert result["connections"] == expected_connections

    def test_unknown_topology_type(self):
        """Test unknown topology type defaults to series"""
        result_json = propose_simple_topology.execute(
            num_resistors=2, topology_type="unknown"
        )
        result = json.loads(result_json)

        # Should default to series
        expected_connections = [["A", "B", "R1"], ["B", "C", "R2"]]
        assert result["connections"] == expected_connections

    def test_single_resistor(self):
        """Test with single resistor"""
        result_json = propose_simple_topology.execute(
            num_resistors=1, topology_type="series"
        )
        result = json.loads(result_json)

        assert len(result["resistors"]) == 1
        assert result["connections"] == [["A", "B", "R1"]]

    def test_zero_resistors(self):
        """Test with zero resistors"""
        result_json = propose_simple_topology.execute(
            num_resistors=0, topology_type="series"
        )
        result = json.loads(result_json)

        assert len(result["resistors"]) == 0
        assert len(result["connections"]) == 0


class TestEstimateResistorValues:
    """Tests for estimate_resistor_values tool"""

    def test_simple_optimization(self):
        """Test resistor value estimation with simple circuit"""
        topology = {
            "resistors": {"R1": 1.0},  # Initial guess
            "connections": [["A", "B", "R1"]],
        }
        measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

        with patch(
            "resistor_network.tools.validate_measurements.execute"
        ) as mock_validate:
            # Mock validation to return decreasing error for value 10
            def mock_validation(topology_str):
                topo = json.loads(topology_str)
                if topo["resistors"]["R1"] == 10.0:
                    return json.dumps({"total_error": 0.0})
                else:
                    return json.dumps({"total_error": 100.0})

            mock_validate.side_effect = mock_validation

            result_json = estimate_resistor_values.execute(
                topology=json.dumps(topology), measurements=json.dumps(measurements)
            )
            result = json.loads(result_json)

        assert result["resistors"]["R1"] == 10.0

    def test_multiple_resistors_optimization(self):
        """Test optimization with multiple resistors"""
        topology = {
            "resistors": {"R1": 1.0, "R2": 1.0, "R3": 1.0},
            "connections": [["A", "B", "R1"], ["B", "C", "R2"], ["C", "D", "R3"]],
        }
        measurements = [{"node_a": "A", "node_b": "D", "resistance": 30.0}]

        with patch(
            "resistor_network.tools.validate_measurements.execute"
        ) as mock_validate:
            mock_validate.return_value = json.dumps(
                {"total_error": 0.0}
            )  # All combinations equally good

            result_json = estimate_resistor_values.execute(
                topology=json.dumps(topology), measurements=json.dumps(measurements)
            )
            result = json.loads(result_json)

        # Should find some optimization (exact values depend on mock behavior)
        assert "resistors" in result
        assert len(result["resistors"]) == 3

    def test_large_circuit_heuristic(self):
        """Test heuristic approach for large circuits (>3 resistors)"""
        topology = {
            "resistors": {f"R{i}": 1.0 for i in range(1, 6)},  # 5 resistors
            "connections": [[f"N{i}", f"N{i+1}", f"R{i}"] for i in range(1, 6)],
        }
        measurements = [{"node_a": "N1", "node_b": "N6", "resistance": 50.0}]

        with patch(
            "resistor_network.tools.validate_measurements.execute"
        ) as mock_validate:
            mock_validate.return_value = json.dumps({"total_error": 10.0})

            result_json = estimate_resistor_values.execute(
                topology=json.dumps(topology), measurements=json.dumps(measurements)
            )
            result = json.loads(result_json)

        assert "resistors" in result
        assert len(result["resistors"]) == 5

    def test_optimization_error_handling(self):
        """Test error handling in optimization"""
        with patch(
            "resistor_network.tools.validate_measurements.execute"
        ) as mock_validate:
            mock_validate.side_effect = Exception("Validation failed")

            result_json = estimate_resistor_values.execute(
                topology="invalid", measurements="invalid"
            )
            result = json.loads(result_json)

        assert "error" in result
        assert "Estimation failed" in result["error"]


class TestGenerateTestMeasurements:
    """Tests for generate_test_measurements tool"""

    def test_simple_measurements_generation(self):
        """Test generation of measurements for simple circuit"""
        topology = {
            "resistors": {"R1": 10.0, "R2": 20.0},
            "connections": [["A", "B", "R1"], ["B", "C", "R2"]],
        }
        terminal_pairs = [["A", "B"], ["B", "C"], ["A", "C"]]

        with patch("resistor_network.tools.simulate_circuit_resistance") as mock_sim:
            mock_sim.side_effect = [10.0, 20.0, 30.0]  # Expected resistances

            result_json = generate_test_measurements.execute(
                topology=json.dumps(topology), terminal_pairs=terminal_pairs
            )
            result = json.loads(result_json)

        assert len(result) == 3
        assert result[0]["node_a"] == "A"
        assert result[0]["node_b"] == "B"
        assert result[0]["resistance"] == 10.0
        assert result[2]["resistance"] == 30.0

    def test_measurement_with_simulation_error(self):
        """Test handling of simulation errors during measurement generation"""
        topology = {"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}
        terminal_pairs = [["X", "Y"]]  # Non-existent nodes

        with patch("resistor_network.tools.simulate_circuit_resistance") as mock_sim:
            mock_sim.side_effect = Exception("Node not found")

            result_json = generate_test_measurements.execute(
                topology=json.dumps(topology), terminal_pairs=terminal_pairs
            )
            result = json.loads(result_json)

        assert len(result) == 1
        assert isinstance(result, list)
        assert len(result) > 0
        assert "error" in result[0]
        assert "Could not measure" in result[0]["error"]

    def test_invalid_terminal_pairs(self):
        """Test handling of invalid terminal pairs"""
        topology = {"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}
        terminal_pairs = [["A"], ["A", "B", "C"]]  # Wrong number of terminals

        result_json = generate_test_measurements.execute(
            topology=json.dumps(topology), terminal_pairs=terminal_pairs
        )
        result = json.loads(result_json)

        # Should skip invalid pairs
        assert len(result) == 0

    def test_rounding_of_results(self):
        """Test that results are properly rounded"""
        topology = {"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}
        terminal_pairs = [["A", "B"]]

        with patch("resistor_network.tools.simulate_circuit_resistance") as mock_sim:
            mock_sim.return_value = 10.123456789

            result_json = generate_test_measurements.execute(
                topology=json.dumps(topology), terminal_pairs=terminal_pairs
            )
            result = json.loads(result_json)

        assert result[0]["resistance"] == 10  # Rounded to 0 decimal places


# Hypothesis-based property tests
class TestCircuitToolsProperties:
    """Property-based tests for circuit analysis tools"""

    @given(
        st.floats(
            min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False
        )
    )
    def test_delta_wye_roundtrip_single_value(self, resistance):
        """Property: Delta->Wye->Delta roundtrip preserves values for equal resistances"""
        # Start with equal delta resistances
        wye_result_json = delta_to_wye_transform.execute(
            ra=resistance, rb=resistance, rc=resistance
        )
        wye_result = json.loads(wye_result_json)
        delta_result_json = wye_to_delta_transform.execute(
            r1=wye_result["r1"], r2=wye_result["r2"], r3=wye_result["r3"]
        )
        delta_result = json.loads(delta_result_json)
        assert abs(delta_result["ra"] - resistance) < 1e-6
        assert abs(delta_result["rb"] - resistance) < 1e-6
        assert abs(delta_result["rc"] - resistance) < 1e-6

    @given(st.integers(min_value=1, max_value=10))
    def test_topology_generation_structure(self, num_resistors):
        """Property: Generated topologies have correct number of resistors"""
        result_json = propose_simple_topology.execute(
            num_resistors=num_resistors, topology_type="series"
        )
        result = json.loads(result_json)

        assert len(result["resistors"]) == num_resistors
        assert len(result["connections"]) == num_resistors

    @given(
        st.floats(
            min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        st.floats(
            min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_series_resistance_simulation(self, r1, r2):
        """Property: Simulated series resistance equals sum of resistances"""
        topology = {
            "resistors": {"R1": r1, "R2": r2},
            "connections": [["A", "B", "R1"], ["B", "C", "R2"]],
        }

        resistance = simulate_circuit_resistance.execute(
            topology=json.dumps(topology), terminal_nodes=["A", "C"]
        )

        assert abs(float(resistance) - (r1 + r2)) < 1e-6

    @given(
        st.floats(
            min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        st.floats(
            min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_parallel_resistance_simulation(self, r1, r2):
        """Property: Simulated parallel resistance follows parallel formula"""
        topology = {
            "resistors": {"R1": r1, "R2": r2},
            "connections": [["A", "B", "R1"], ["A", "B", "R2"]],
        }

        resistance = simulate_circuit_resistance.execute(
            topology=json.dumps(topology), terminal_nodes=["A", "B"]
        )

        expected = (r1 * r2) / (r1 + r2)
        assert abs(float(resistance) - expected) < 1e-6
