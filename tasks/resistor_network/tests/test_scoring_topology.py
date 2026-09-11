import json

import pytest

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import composite
from resistor_network.score import (
    _score_resistor_values,
    _score_topology_structure,
    check_resistor_topology,
)

# ============================================================================
# TEST CLASSES
# ============================================================================


class TestTopologyStructureScoring:
    """Test the binary topology structure scoring function"""

    def test_perfect_match(self):
        """Test when connections match exactly"""
        expected = [("A", "B", "R1"), ("B", "C", "R2")]
        proposed = [("A", "B", "R1"), ("B", "C", "R2")]
        assert _score_topology_structure(proposed, expected) == 1.0

    def test_order_independence(self):
        """Test that connection order doesn't matter"""
        expected = [("A", "B", "R1"), ("B", "C", "R2")]
        proposed = [("B", "C", "R2"), ("A", "B", "R1")]  # Different order
        assert _score_topology_structure(proposed, expected) == 1.0

    def test_node_order_independence(self):
        """Test that node order in connection doesn't matter"""
        expected = [("A", "B", "R1")]
        proposed = [("B", "A", "R1")]  # Nodes swapped
        assert _score_topology_structure(proposed, expected) == 1.0

    def test_partial_match_binary(self):
        """Test that partial matches return 0.0 (binary scoring)"""
        expected = [("A", "B", "R1"), ("B", "C", "R2"), ("C", "D", "R3")]
        proposed = [("A", "B", "R1"), ("B", "C", "R2")]  # Missing one connection
        assert _score_topology_structure(proposed, expected) == 0.0

    def test_no_match(self):
        """Test when no connections match"""
        expected = [("A", "B", "R1"), ("B", "C", "R2")]
        proposed = [("X", "Y", "R3"), ("Y", "Z", "R4")]
        assert _score_topology_structure(proposed, expected) == 0.0

    def test_different_lengths(self):
        """Test when connection lists have different lengths"""
        expected = [("A", "B", "R1")]
        proposed = [("A", "B", "R1"), ("B", "C", "R2")]
        assert _score_topology_structure(proposed, expected) == 0.0

    def test_empty_lists(self):
        """Test with empty connection lists"""
        assert _score_topology_structure([], []) == 1.0
        assert _score_topology_structure([("A", "B", "R1")], []) == 0.0
        assert _score_topology_structure([], [("A", "B", "R1")]) == 0.0

    def test_single_connection_mismatch(self):
        """Test that even a single wrong connection causes failure"""
        expected = [("A", "B", "R1"), ("B", "C", "R2")]
        proposed = [("A", "B", "R1"), ("B", "D", "R2")]  # One connection wrong
        assert _score_topology_structure(proposed, expected) == 0.0


class TestResistorValueScoring:
    """Test the binary resistor value scoring function"""

    def test_perfect_match(self):
        """Test when resistor values match exactly"""
        expected = {"R1": 100.0, "R2": 200.0}
        proposed = {"R1": 100.0, "R2": 200.0}
        assert _score_resistor_values(proposed, expected, 0.1) == 1.0

    def test_all_within_tolerance(self):
        """Test when all values are within tolerance"""
        expected = {"R1": 100.0, "R2": 200.0}
        proposed = {"R1": 105.0, "R2": 195.0}  # Both within 10% tolerance
        assert _score_resistor_values(proposed, expected, 0.1) == 1.0

    def test_one_outside_tolerance_binary(self):
        """Test that one resistor outside tolerance causes complete failure"""
        expected = {"R1": 100.0, "R2": 200.0}
        proposed = {"R1": 100.0, "R2": 240.0}  # R2 is 20% off (outside 10% tolerance)
        assert (
            _score_resistor_values(proposed, expected, 0.1) == 0.0
        )  # Binary: complete failure

    def test_boundary_cases(self):
        """Test values exactly at tolerance boundary"""
        expected = {"R1": 100.0}

        # Exactly at tolerance (10%)
        proposed_at_boundary = {"R1": 110.0}
        assert _score_resistor_values(proposed_at_boundary, expected, 0.1) == 1.0

        # Just outside tolerance
        proposed_outside = {"R1": 110.1}
        assert _score_resistor_values(proposed_outside, expected, 0.1) == 0.0

    def test_zero_expected_value(self):
        """Test handling of zero expected values"""
        expected = {"R1": 0.0}
        proposed_correct = {"R1": 0.0}
        proposed_wrong = {"R1": 10.0}

        assert _score_resistor_values(proposed_correct, expected, 0.1) == 1.0
        assert _score_resistor_values(proposed_wrong, expected, 0.1) == 0.0

    def test_different_resistor_sets(self):
        """Test when resistor sets don't match"""
        expected = {"R1": 100.0, "R2": 200.0}
        proposed = {"R1": 100.0, "R3": 300.0}  # Different resistor names
        assert _score_resistor_values(proposed, expected, 0.1) == 0.0

    def test_empty_dicts(self):
        """Test with empty resistor dictionaries"""
        # Empty dicts should return 1.0 - if no resistors to check, all are "within tolerance"
        # This is logically consistent: ALL (zero) resistors are within the specified tolerance
        assert _score_resistor_values({}, {}, 0.1) == 1.0

    def test_tight_tolerance(self):
        """Test with very tight tolerance"""
        expected = {"R1": 100.0}
        proposed = {"R1": 100.5}  # 0.5% error

        # Should pass with 1% tolerance
        assert _score_resistor_values(proposed, expected, 0.01) == 1.0

        # Should fail with 0.1% tolerance
        assert _score_resistor_values(proposed, expected, 0.001) == 0.0


class TestMainTopologyChecker:
    """Test the main check_resistor_topology function"""

    def setup_method(self):
        """Set up test data"""
        self.expected_topology = {
            "resistors": {"R1": 100.0, "R2": 200.0},
            "connections": [("A", "B", "R1"), ("B", "C", "R2")],
        }

    def test_functional_scoring_accepts_smaller_equivalent_submission(self):
        expected = {
            "resistors": {"R1": 10.0, "R2": 20.0, "R3": 20.0},
            "connections": [
                ["A", "B", "R1"],
                ["B", "C", "R2"],
                ["B", "C", "R3"],
            ],
        }
        checker = check_resistor_topology(
            expected_topology=expected,
            expected_measurements=[
                {"node_a": "A", "node_b": "B", "resistance": 10.0},
                {"node_a": "B", "node_b": "C", "resistance": 10.0},
                {"node_a": "A", "node_b": "C", "resistance": 20.0},
            ],
            use_functional_scoring=True,
            topology_weight=0.0,
            functional_weight=1.0,
            exact_values_weight=0.0,
        )
        one_resistor = json.dumps(
            {
                "resistors": {"R": 10.0, "S": 10.0},
                "connections": [["A", "B", "R"], ["B", "C", "S"]],
            }
        )
        assert checker(one_resistor) == 1.0

    def test_json_string_input_perfect(self):
        """Test with JSON string input - perfect match"""
        checker = check_resistor_topology(
            expected_topology=self.expected_topology,
            use_functional_scoring=False,
            topology_weight=0.5,
            functional_weight=0.0,
            exact_values_weight=0.5,
            require_both=True,
        )

        proposed_json = json.dumps(
            {
                "resistors": {"R1": 100.0, "R2": 200.0},
                "connections": [("A", "B", "R1"), ("B", "C", "R2")],
            }
        )

        assert checker(proposed_json) == 1.0

    def test_require_both_true_topology_wrong(self):
        """Test require_both=True with wrong topology"""
        checker = check_resistor_topology(
            expected_topology=self.expected_topology,
            use_functional_scoring=False,
            topology_weight=0.5,
            functional_weight=0.0,
            exact_values_weight=0.5,
            require_both=True,
        )

        proposed_json = json.dumps(
            {
                "resistors": {"R1": 100.0, "R2": 200.0},  # Resistors correct
                "connections": [("A", "X", "R1"), ("B", "C", "R2")],  # Topology wrong
            }
        )

        assert checker(proposed_json) == 0.0  # Both must be perfect

    def test_require_both_true_resistors_wrong(self):
        """Test require_both=True with wrong resistors"""
        checker = check_resistor_topology(
            expected_topology=self.expected_topology,
            use_functional_scoring=False,
            topology_weight=0.5,
            functional_weight=0.0,
            exact_values_weight=0.5,
            tolerance=0.05,
            require_both=True,
        )

        proposed_json = json.dumps(
            {
                "resistors": {"R1": 120.0, "R2": 200.0},  # R1 outside 5% tolerance
                "connections": [("A", "B", "R1"), ("B", "C", "R2")],  # Topology correct
            }
        )

        assert checker(proposed_json) == 0.0  # Both must be perfect

    def test_require_both_false_topology_correct(self):
        """Test require_both=False with correct topology, wrong resistors"""
        checker = check_resistor_topology(
            expected_topology=self.expected_topology,
            use_functional_scoring=False,
            topology_weight=0.5,
            functional_weight=0.0,
            exact_values_weight=0.5,
            tolerance=0.05,
            require_both=False,
        )

        proposed_json = json.dumps(
            {
                "resistors": {"R1": 120.0, "R2": 200.0},  # R1 outside tolerance
                "connections": [("A", "B", "R1"), ("B", "C", "R2")],  # Topology correct
            }
        )

        assert checker(proposed_json) == 1.0  # Topology is perfect, that's enough

    def test_require_both_false_resistors_correct(self):
        """Test require_both=False with correct resistors, wrong topology"""
        checker = check_resistor_topology(
            expected_topology=self.expected_topology,
            use_functional_scoring=False,
            topology_weight=0.5,
            functional_weight=0.0,
            exact_values_weight=0.5,
            require_both=False,
        )

        proposed_json = json.dumps(
            {
                "resistors": {"R1": 100.0, "R2": 200.0},  # Resistors correct
                "connections": [("A", "X", "R1"), ("B", "C", "R2")],  # Topology wrong
            }
        )

        assert checker(proposed_json) == 1.0  # Resistors are perfect, that's enough

    def test_require_both_false_both_wrong(self):
        """Test require_both=False with both wrong"""
        checker = check_resistor_topology(
            expected_topology=self.expected_topology,
            use_functional_scoring=False,
            topology_weight=0.5,
            functional_weight=0.0,
            exact_values_weight=0.5,
            tolerance=0.05,
            require_both=False,
        )

        proposed_json = json.dumps(
            {
                "resistors": {"R1": 120.0, "R2": 200.0},  # Resistors wrong
                "connections": [("A", "X", "R1"), ("B", "C", "R2")],  # Topology wrong
            }
        )

        assert checker(proposed_json) == 0.0  # Neither is perfect

    def test_malformed_json(self):
        """Test with malformed JSON"""
        checker = check_resistor_topology(
            expected_topology=self.expected_topology,
            use_functional_scoring=False,
            topology_weight=0.5,
            functional_weight=0.0,
            exact_values_weight=0.5,
        )
        assert checker("{ invalid json }") == 0.0

    def test_missing_required_keys(self):
        """Test with missing required keys"""
        checker = check_resistor_topology(
            expected_topology=self.expected_topology,
            use_functional_scoring=False,
            topology_weight=0.5,
            functional_weight=0.0,
            exact_values_weight=0.5,
        )

        # Missing 'connections'
        incomplete_json = json.dumps({"resistors": {"R1": 100.0}})
        assert checker(incomplete_json) == 0.0

        # Missing 'resistors'
        incomplete_json = json.dumps({"connections": [("A", "B", "R1")]})
        assert checker(incomplete_json) == 0.0

    def test_empty_topology_data(self):
        """Test with empty topology data"""
        checker = check_resistor_topology(
            expected_topology=self.expected_topology,
            use_functional_scoring=False,
            topology_weight=0.5,
            functional_weight=0.0,
            exact_values_weight=0.5,
        )
        assert checker("null") == 0.0
        assert checker("{}") == 0.0


# ============================================================================
# HYPOTHESIS PROPERTY-BASED TESTS
# ============================================================================


@composite
def resistor_dict(draw, min_resistors=1, max_resistors=5):
    """Generate a dictionary of resistor names and values"""
    resistor_count = draw(st.integers(min_value=min_resistors, max_value=max_resistors))
    resistor_names = [f"R{i}" for i in range(1, resistor_count + 1)]
    resistor_values = draw(
        st.lists(
            st.floats(min_value=0.1, max_value=1000.0, exclude_min=True),
            min_size=resistor_count,
            max_size=resistor_count,
        )
    )
    return dict(zip(resistor_names, resistor_values, strict=False))


@composite
def connection_list(draw, resistor_names):
    """Generate a list of connections using given resistor names"""
    if not resistor_names:
        return []

    node_names = ["A", "B", "C", "D", "E", "F"]
    connections = []

    for resistor in resistor_names:
        node1 = draw(st.sampled_from(node_names))
        node2 = draw(st.sampled_from([n for n in node_names if n != node1]))
        connections.append((node1, node2, resistor))

    return connections


class TestPropertyBasedTests:
    """Property-based tests using Hypothesis for binary scoring"""

    @given(
        st.lists(
            st.tuples(st.text(min_size=1), st.text(min_size=1), st.text(min_size=1)),
            min_size=0,
            max_size=10,
        )
    )
    def test_topology_score_binary_range(self, connections):
        """Property: binary topology scores should be exactly 0.0 or 1.0"""
        score = _score_topology_structure(connections, connections)
        assert score in [0.0, 1.0]

        # Perfect match should give score 1.0
        assert score == 1.0

    @given(resistor_dict(), st.floats(min_value=0.01, max_value=1.0))
    def test_resistor_score_binary_range(self, resistors, tolerance):
        """Property: binary resistor scores should be exactly 0.0 or 1.0"""
        score = _score_resistor_values(resistors, resistors, tolerance)
        assert score in [0.0, 1.0]

        # Perfect match should give score 1.0
        assert score == 1.0

    @given(resistor_dict(min_resistors=1, max_resistors=3))
    def test_single_resistor_failure_causes_total_failure(self, expected_resistors):
        """Property: if any resistor is outside tolerance, total score should be 0.0"""
        tolerance = 0.1  # 10% tolerance

        # Create version where first resistor is way outside tolerance
        first_resistor = next(iter(expected_resistors.keys()))
        proposed_resistors = expected_resistors.copy()
        proposed_resistors[first_resistor] = (
            expected_resistors[first_resistor] * 2.0
        )  # 100% error

        score = _score_resistor_values(
            proposed_resistors, expected_resistors, tolerance
        )
        assert score == 0.0  # Should fail completely

    @given(st.text())
    def test_topology_checker_robustness(self, random_input):
        """Property: topology checker should never crash with random input"""
        expected_topology = {
            "resistors": {"R1": 100.0},
            "connections": [("A", "B", "R1")],
        }

        checker = check_resistor_topology(
            expected_topology=expected_topology,
            use_functional_scoring=False,
            topology_weight=0.5,
            functional_weight=0.0,
            exact_values_weight=0.5,
        )

        # Should not crash, should return a valid binary score
        try:
            score = checker(random_input)
            assert score in [0.0, 1.0]  # Binary scoring only
        except Exception:
            # If it crashes, that's a problem
            pytest.fail("Topology checker crashed with random input")

    @given(
        resistor_dict(min_resistors=1, max_resistors=3),
        st.floats(min_value=0.01, max_value=0.5),
    )
    def test_tolerance_effect_binary(self, resistors, tolerance):
        """Property: binary scoring should show tolerance effect at boundaries

        Note: We avoid testing exactly at tolerance boundaries due to floating point
        precision issues. Instead, we test well within bounds and clearly outside bounds.
        """
        if not resistors:
            return

        # Create version with error slightly inside tolerance boundary to avoid floating point precision issues
        first_resistor = next(iter(resistors.keys()))
        test_resistors = resistors.copy()
        # Use 95% of tolerance to avoid floating point precision issues at exact boundary
        test_resistors[first_resistor] = resistors[first_resistor] * (
            1 + tolerance * 0.95
        )

        # Should pass with the given tolerance (well within bounds)
        score_at_tolerance = _score_resistor_values(
            test_resistors, resistors, tolerance
        )
        assert score_at_tolerance == 1.0

        # Create version that's clearly outside tolerance
        far_test_resistors = resistors.copy()
        far_test_resistors[first_resistor] = resistors[first_resistor] * (
            1 + tolerance * 1.5
        )  # 150% of tolerance

        # Should fail when clearly outside tolerance
        score_outside = _score_resistor_values(far_test_resistors, resistors, tolerance)
        assert score_outside == 0.0

    @given(
        st.lists(
            st.tuples(st.text(min_size=1), st.text(min_size=1), st.text(min_size=1)),
            min_size=1,
            max_size=5,
        )
    )
    def test_single_connection_failure_causes_total_failure(self, connections):
        """Property: if any connection is wrong, topology score should be 0.0"""
        # Create a version with one connection changed
        modified_connections = connections.copy()
        if modified_connections:
            # Change the first connection's resistor name
            old_conn = modified_connections[0]
            modified_connections[0] = (old_conn[0], old_conn[1], old_conn[2] + "_WRONG")

            score = _score_topology_structure(modified_connections, connections)
            assert score == 0.0  # Should fail completely


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for binary scoring system"""

    def test_realistic_circuit_scenario_binary(self):
        """Test a realistic circuit analysis scenario with binary scoring"""
        # Simple voltage divider circuit
        expected_topology = {
            "resistors": {
                "R1": 1000.0,  # 1kΩ
                "R2": 2000.0,  # 2kΩ
            },
            "connections": [
                ("VCC", "OUT", "R1"),  # R1 from VCC to output
                ("OUT", "GND", "R2"),  # R2 from output to ground
            ],
        }

        checker_both = check_resistor_topology(
            expected_topology=expected_topology,
            use_functional_scoring=False,
            topology_weight=0.5,
            functional_weight=0.0,
            exact_values_weight=0.5,
            tolerance=0.05,
            require_both=True,
        )
        checker_either = check_resistor_topology(
            expected_topology=expected_topology,
            use_functional_scoring=False,
            topology_weight=0.5,
            functional_weight=0.0,
            exact_values_weight=0.5,
            tolerance=0.05,
            require_both=False,
        )

        # Test cases for binary scoring
        test_cases = [
            # Perfect match
            {
                "input": {
                    "resistors": {"R1": 1000.0, "R2": 2000.0},
                    "connections": [("VCC", "OUT", "R1"), ("OUT", "GND", "R2")],
                },
                "expected_both": 1.0,
                "expected_either": 1.0,
                "description": "Perfect match",
            },
            # Good resistor values, perfect topology
            {
                "input": {
                    "resistors": {"R1": 1020.0, "R2": 1980.0},  # Within 5% tolerance
                    "connections": [("VCC", "OUT", "R1"), ("OUT", "GND", "R2")],
                },
                "expected_both": 1.0,
                "expected_either": 1.0,
                "description": "Good resistors and topology",
            },
            # Bad resistor values, perfect topology
            {
                "input": {
                    "resistors": {
                        "R1": 1200.0,
                        "R2": 2000.0,
                    },  # R1 outside 5% tolerance
                    "connections": [("VCC", "OUT", "R1"), ("OUT", "GND", "R2")],
                },
                "expected_both": 0.0,  # Resistors failed, need both
                "expected_either": 1.0,  # Topology perfect, either is enough
                "description": "Bad resistors, good topology",
            },
            # Good resistor values, bad topology
            {
                "input": {
                    "resistors": {"R1": 1000.0, "R2": 2000.0},
                    "connections": [
                        ("VCC", "GND", "R1"),
                        ("VCC", "OUT", "R2"),
                    ],  # Wrong connections
                },
                "expected_both": 0.0,  # Topology failed, need both
                "expected_either": 1.0,  # Resistors perfect, either is enough
                "description": "Good resistors, bad topology",
            },
            # Both wrong
            {
                "input": {
                    "resistors": {"R1": 1200.0, "R2": 2000.0},  # R1 outside tolerance
                    "connections": [
                        ("VCC", "GND", "R1"),
                        ("VCC", "OUT", "R2"),
                    ],  # Wrong connections
                },
                "expected_both": 0.0,
                "expected_either": 0.0,
                "description": "Both resistors and topology wrong",
            },
        ]

        for case in test_cases:
            json_input = json.dumps(case["input"])

            score_both = checker_both(json_input)
            score_either = checker_either(json_input)

            assert (
                score_both == case["expected_both"]
            ), f"require_both=True failed for: {case['description']}"
            assert (
                score_either == case["expected_either"]
            ), f"require_both=False failed for: {case['description']}"

    def test_complex_circuit_binary(self):
        """Test with more complex circuit topology"""
        # Bridge circuit
        expected_topology = {
            "resistors": {
                "R1": 100.0,
                "R2": 200.0,
                "R3": 300.0,
                "R4": 400.0,
                "R5": 500.0,
            },
            "connections": [
                ("A", "B", "R1"),
                ("A", "C", "R2"),
                ("B", "D", "R3"),
                ("C", "D", "R4"),
                ("B", "C", "R5"),  # Bridge resistor
            ],
        }

        checker = check_resistor_topology(
            expected_topology=expected_topology,
            use_functional_scoring=False,
            topology_weight=0.5,
            functional_weight=0.0,
            exact_values_weight=0.5,
            tolerance=0.1,
            require_both=True,
        )

        # Perfect match should work
        perfect_json = json.dumps(
            {
                "resistors": {
                    "R1": 100.0,
                    "R2": 200.0,
                    "R3": 300.0,
                    "R4": 400.0,
                    "R5": 500.0,
                },
                "connections": [
                    ("A", "B", "R1"),
                    ("A", "C", "R2"),
                    ("B", "D", "R3"),
                    ("C", "D", "R4"),
                    ("B", "C", "R5"),
                ],
            }
        )
        assert checker(perfect_json) == 1.0

        # Missing one connection should fail
        incomplete_json = json.dumps(
            {
                "resistors": {
                    "R1": 100.0,
                    "R2": 200.0,
                    "R3": 300.0,
                    "R4": 400.0,
                    "R5": 500.0,
                },
                "connections": [
                    ("A", "B", "R1"),
                    ("A", "C", "R2"),
                    ("B", "D", "R3"),
                    ("C", "D", "R4"),  # Missing R5 connection
                ],
            }
        )
        assert checker(incomplete_json) == 0.0


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions for binary scoring"""

    def test_very_large_numbers(self):
        """Test with very large resistor values"""
        large_resistors = {"R1": 1e12, "R2": 1e15}  # TΩ range
        score = _score_resistor_values(large_resistors, large_resistors, 0.1)
        assert score == 1.0

    def test_very_small_numbers(self):
        """Test with very small resistor values"""
        small_resistors = {"R1": 1e-6, "R2": 1e-9}  # μΩ to nΩ range
        score = _score_resistor_values(small_resistors, small_resistors, 0.1)
        assert score == 1.0

    def test_unicode_in_names(self):
        """Test with Unicode characters in resistor/node names"""
        expected = [("Ω", "μ", "R₁")]
        proposed = [("Ω", "μ", "R₁")]
        assert _score_topology_structure(proposed, expected) == 1.0

    def test_very_long_connection_lists(self):
        """Test with large numbers of connections"""
        n = 100
        connections = [(f"node_{i}", f"node_{i + 1}", f"R_{i}") for i in range(n)]
        score = _score_topology_structure(connections, connections)
        assert score == 1.0

        # One wrong connection should fail everything
        wrong_connections = connections.copy()
        wrong_connections[50] = ("WRONG", "WRONG", "WRONG")
        score = _score_topology_structure(wrong_connections, connections)
        assert score == 0.0

    def test_extreme_tolerances(self):
        """Test with extreme tolerance values"""
        resistors = {"R1": 100.0}
        imperfect = {"R1": 110.0}  # 10% error

        # Very tight tolerance - should fail
        score_tight = _score_resistor_values(
            imperfect, resistors, 0.001
        )  # 0.1% tolerance
        assert score_tight == 0.0

        # Very loose tolerance - should pass
        score_loose = _score_resistor_values(
            imperfect, resistors, 1.0
        )  # 100% tolerance
        assert score_loose == 1.0

    def test_boundary_precision(self):
        """Test precision at tolerance boundaries with clean numbers"""
        expected = {"R1": 100.0}  # Use clean integer-based values
        tolerance = 0.1  # 10%

        # Test values very close to boundary
        test_cases = [
            (109.0, 1.0),  # 9% error - should pass
            (110.0, 1.0),  # Exactly 10% error - should pass
            (111.0, 0.0),  # 11% error - should fail
            (90.0, 1.0),  # -10% error - should pass
            (89.0, 0.0),  # -11% error - should fail
        ]

        for value, expected_score in test_cases:
            proposed = {"R1": value}
            score = _score_resistor_values(proposed, expected, tolerance)
            relative_error = abs(value - 100.0) / 100.0
            assert score == expected_score, (
                f"Failed for R1={value}: relative_error={relative_error:.3f}, "
                f"tolerance={tolerance}, expected_score={expected_score}, actual_score={score}"
            )
