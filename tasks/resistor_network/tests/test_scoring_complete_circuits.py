import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import composite

os.environ["CORRAL_WORK_DIR"] = str(Path(__file__).parent / "test_files" / "temp")
from resistor_network.score import check_complete_circuit_solution


class TestCheckCompleteCircuitSolution:
    """Unit tests for check_complete_circuit_solution function"""

    def test_returns_callable(self):
        """Test that function returns a callable"""
        expected_topology = {
            "resistors": {"R1": 10.0},
            "connections": [["A", "B", "R1"]],
        }
        expected_measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

        scorer = check_complete_circuit_solution(
            expected_topology, expected_measurements
        )
        assert callable(scorer)

    @patch("resistor_network.score.check_resistor_topology")
    @patch("resistor_network.score.check_resistance_measurements")
    def test_default_weights(self, mock_measurement_scorer, mock_topology_scorer):
        """Test default weight distribution (0.6 topology, 0.4 measurements)"""
        # Setup mocks
        mock_topology_score_fn = MagicMock(return_value=0.8)
        mock_measurement_score_fn = MagicMock(return_value=0.6)
        mock_topology_scorer.return_value = mock_topology_score_fn
        mock_measurement_scorer.return_value = mock_measurement_score_fn

        expected_topology = {
            "resistors": {"R1": 10.0},
            "connections": [["A", "B", "R1"]],
        }
        expected_measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

        scorer = check_complete_circuit_solution(
            expected_topology, expected_measurements
        )

        topology_input = (
            '{"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}'
        )
        score = scorer(topology_input)

        # Expected: 0.6 * 0.8 + 0.4 * 0.6 = 0.48 + 0.24 = 0.72
        expected_score = 0.6 * 0.8 + 0.4 * 0.6
        assert abs(score - expected_score) < 1e-6

        # Verify both scorers were called
        mock_topology_score_fn.assert_called_once_with(topology_input)
        mock_measurement_score_fn.assert_called_once_with(topology_input)

    @patch("resistor_network.score.check_resistor_topology")
    @patch("resistor_network.score.check_resistance_measurements")
    def test_custom_weights(self, mock_measurement_scorer, mock_topology_scorer):
        """Test custom weight distribution"""
        # Setup mocks
        mock_topology_score_fn = MagicMock(return_value=0.9)
        mock_measurement_score_fn = MagicMock(return_value=0.7)
        mock_topology_scorer.return_value = mock_topology_score_fn
        mock_measurement_scorer.return_value = mock_measurement_score_fn

        expected_topology = {
            "resistors": {"R1": 10.0},
            "connections": [["A", "B", "R1"]],
        }
        expected_measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

        # Custom weights: 80% topology, 20% measurements
        scorer = check_complete_circuit_solution(
            expected_topology,
            expected_measurements,
            topology_weight=0.8,
            measurement_weight=0.2,
        )

        topology_input = (
            '{"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}'
        )
        score = scorer(topology_input)

        # Expected: 0.8 * 0.9 + 0.2 * 0.7 = 0.72 + 0.14 = 0.86
        expected_score = 0.8 * 0.9 + 0.2 * 0.7
        assert abs(score - expected_score) < 1e-6

    @patch("resistor_network.score.check_resistor_topology")
    @patch("resistor_network.score.check_resistance_measurements")
    def test_perfect_scores(self, mock_measurement_scorer, mock_topology_scorer):
        """Test when both scorers return perfect scores"""
        # Setup mocks for perfect scores
        mock_topology_score_fn = MagicMock(return_value=1.0)
        mock_measurement_score_fn = MagicMock(return_value=1.0)
        mock_topology_scorer.return_value = mock_topology_score_fn
        mock_measurement_scorer.return_value = mock_measurement_score_fn

        expected_topology = {
            "resistors": {"R1": 10.0},
            "connections": [["A", "B", "R1"]],
        }
        expected_measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

        scorer = check_complete_circuit_solution(
            expected_topology, expected_measurements
        )

        topology_input = (
            '{"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}'
        )
        score = scorer(topology_input)

        # Perfect score should be 1.0 regardless of weights
        assert abs(score - 1.0) < 1e-6

    @patch("resistor_network.score.check_resistor_topology")
    @patch("resistor_network.score.check_resistance_measurements")
    def test_zero_scores(self, mock_measurement_scorer, mock_topology_scorer):
        """Test when both scorers return zero scores"""
        # Setup mocks for zero scores
        mock_topology_score_fn = MagicMock(return_value=0.0)
        mock_measurement_score_fn = MagicMock(return_value=0.0)
        mock_topology_scorer.return_value = mock_topology_score_fn
        mock_measurement_scorer.return_value = mock_measurement_score_fn

        expected_topology = {
            "resistors": {"R1": 10.0},
            "connections": [["A", "B", "R1"]],
        }
        expected_measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

        scorer = check_complete_circuit_solution(
            expected_topology, expected_measurements
        )

        topology_input = (
            '{"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}'
        )
        score = scorer(topology_input)

        # Zero scores should result in 0.0
        assert score == 0.0

    @patch("resistor_network.score.check_resistor_topology")
    @patch("resistor_network.score.check_resistance_measurements")
    def test_mixed_scores(self, mock_measurement_scorer, mock_topology_scorer):
        """Test mixed scores (high topology, low measurements)"""
        # Setup mocks
        mock_topology_score_fn = MagicMock(return_value=1.0)
        mock_measurement_score_fn = MagicMock(return_value=0.0)
        mock_topology_scorer.return_value = mock_topology_score_fn
        mock_measurement_scorer.return_value = mock_measurement_score_fn

        expected_topology = {
            "resistors": {"R1": 10.0},
            "connections": [["A", "B", "R1"]],
        }
        expected_measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

        scorer = check_complete_circuit_solution(
            expected_topology, expected_measurements
        )

        topology_input = (
            '{"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}'
        )
        score = scorer(topology_input)

        # Expected: 0.6 * 1.0 + 0.4 * 0.0 = 0.6
        expected_score = 0.6
        assert abs(score - expected_score) < 1e-6

    @patch("resistor_network.score.check_resistor_topology")
    @patch("resistor_network.score.check_resistance_measurements")
    def test_weights_sum_to_one(self, mock_measurement_scorer, mock_topology_scorer):
        """Test that weights should ideally sum to 1.0"""
        # Setup mocks
        mock_topology_score_fn = MagicMock(return_value=0.5)
        mock_measurement_score_fn = MagicMock(return_value=0.5)
        mock_topology_scorer.return_value = mock_topology_score_fn
        mock_measurement_scorer.return_value = mock_measurement_score_fn

        expected_topology = {
            "resistors": {"R1": 10.0},
            "connections": [["A", "B", "R1"]],
        }
        expected_measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

        # Test weights that sum to 1.0
        scorer1 = check_complete_circuit_solution(
            expected_topology,
            expected_measurements,
            topology_weight=0.7,
            measurement_weight=0.3,
        )

        # Test weights that don't sum to 1.0
        scorer2 = check_complete_circuit_solution(
            expected_topology,
            expected_measurements,
            topology_weight=0.8,
            measurement_weight=0.8,  # Sum = 1.6
        )

        topology_input = (
            '{"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}'
        )

        score1 = scorer1(topology_input)
        score2 = scorer2(topology_input)

        # score1 should be 0.7 * 0.5 + 0.3 * 0.5 = 0.5
        # score2 should be 0.8 * 0.5 + 0.8 * 0.5 = 0.8
        assert abs(score1 - 0.5) < 1e-6
        assert abs(score2 - 0.8) < 1e-6

        # The function allows any weights, doesn't enforce sum = 1

    @patch("resistor_network.score.check_resistor_topology")
    @patch("resistor_network.score.check_resistance_measurements")
    def test_tolerance_parameter_passed(
        self, mock_measurement_scorer, mock_topology_scorer
    ):
        """Test that tolerance parameter is passed to both scorers"""
        mock_topology_scorer.return_value = MagicMock(return_value=0.5)
        mock_measurement_scorer.return_value = MagicMock(return_value=0.5)

        expected_topology = {
            "resistors": {"R1": 10.0},
            "connections": [["A", "B", "R1"]],
        }
        expected_measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

        custom_tolerance = 0.15
        _scorer = check_complete_circuit_solution(
            expected_topology, expected_measurements, tolerance=custom_tolerance
        )

        # Verify that both scorers were created with the custom tolerance
        mock_topology_scorer.assert_called_once_with(
            expected_topology=expected_topology,
            use_functional_scoring=False,
            topology_weight=0.5,
            functional_weight=0.0,
            exact_values_weight=0.5,
            tolerance=custom_tolerance,
            require_both=True,
        )
        mock_measurement_scorer.assert_called_once_with(
            expected_measurements, custom_tolerance
        )

    @patch("resistor_network.score.check_resistor_topology")
    @patch("resistor_network.score.check_resistance_measurements")
    def test_scorer_exception_handling(
        self, mock_measurement_scorer, mock_topology_scorer
    ):
        """Test handling of exceptions from topology scorer"""
        # Make topology scorer raise an exception
        mock_topology_score_fn = MagicMock(
            side_effect=Exception("Topology scorer failed")
        )
        mock_measurement_score_fn = MagicMock(return_value=0.8)
        mock_topology_scorer.return_value = mock_topology_score_fn
        mock_measurement_scorer.return_value = mock_measurement_score_fn

        expected_topology = {
            "resistors": {"R1": 10.0},
            "connections": [["A", "B", "R1"]],
        }
        expected_measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

        scorer = check_complete_circuit_solution(
            expected_topology, expected_measurements
        )

        topology_input = (
            '{"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}'
        )
        score = scorer(topology_input)

        # Should return 0.0 when exception occurs
        assert score == 0.0

    @patch("resistor_network.score.check_resistor_topology")
    @patch("resistor_network.score.check_resistance_measurements")
    def test_measurement_scorer_exception_handling(
        self, mock_measurement_scorer, mock_topology_scorer
    ):
        """Test handling of exceptions from measurement scorer"""
        # Make measurement scorer raise an exception
        mock_topology_score_fn = MagicMock(return_value=0.8)
        mock_measurement_score_fn = MagicMock(
            side_effect=Exception("Measurement scorer failed")
        )
        mock_topology_scorer.return_value = mock_topology_score_fn
        mock_measurement_scorer.return_value = mock_measurement_score_fn

        expected_topology = {
            "resistors": {"R1": 10.0},
            "connections": [["A", "B", "R1"]],
        }
        expected_measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

        scorer = check_complete_circuit_solution(
            expected_topology, expected_measurements
        )

        topology_input = (
            '{"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}'
        )
        score = scorer(topology_input)

        # Should return 0.0 when exception occurs
        assert score == 0.0

    @patch("resistor_network.score.check_resistor_topology")
    @patch("resistor_network.score.check_resistance_measurements")
    def test_empty_topology_and_measurements(
        self, mock_measurement_scorer, mock_topology_scorer
    ):
        """Test with empty topology and measurements"""
        mock_topology_scorer.return_value = MagicMock(return_value=0.0)
        mock_measurement_scorer.return_value = MagicMock(return_value=0.0)

        expected_topology = {"resistors": {}, "connections": []}
        expected_measurements = []

        scorer = check_complete_circuit_solution(
            expected_topology, expected_measurements
        )

        topology_input = '{"resistors": {}, "connections": []}'
        score = scorer(topology_input)

        # Should handle empty inputs gracefully
        assert score == 0.0

    def test_score_bounds(self):
        """Test that scores are always within valid bounds [0.0, 1.0]"""
        # This is more of an integration test to ensure the weighted combination
        # doesn't produce invalid scores
        with (
            patch("resistor_network.score.check_resistor_topology") as mock_topo,
            patch("resistor_network.score.check_resistance_measurements") as mock_meas,
        ):
            # Test various score combinations
            test_cases = [
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.5, 0.5),
                (0.3, 0.7),
                (0.9, 0.1),
            ]

            for topo_score, meas_score in test_cases:
                mock_topo.return_value = MagicMock(return_value=topo_score)
                mock_meas.return_value = MagicMock(return_value=meas_score)

                expected_topology = {
                    "resistors": {"R1": 10.0},
                    "connections": [["A", "B", "R1"]],
                }
                expected_measurements = [
                    {"node_a": "A", "node_b": "B", "resistance": 10.0}
                ]

                scorer = check_complete_circuit_solution(
                    expected_topology, expected_measurements
                )

                topology_input = (
                    '{"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}'
                )
                score = scorer(topology_input)

                # Score should always be between 0.0 and 1.0
                assert 0.0 <= score <= 1.0


# Hypothesis-based property tests
@composite
def weight_pairs(draw):
    """Generate valid weight pairs"""
    topology_weight = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    measurement_weight = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    return topology_weight, measurement_weight


@composite
def score_pairs(draw):
    """Generate valid score pairs"""
    topology_score = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    measurement_score = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    return topology_score, measurement_score


class TestCompleteCircuitHypothesis:
    """Property-based tests using Hypothesis"""

    @given(weight_pairs(), score_pairs())
    def test_weighted_average_property(self, weights, scores):
        """Property: Final score should be weighted average of component scores"""
        topology_weight, measurement_weight = weights
        topology_score, measurement_score = scores

        with (
            patch(
                "resistor_network.score.check_resistor_topology"
            ) as mock_topology_scorer,
            patch(
                "resistor_network.score.check_resistance_measurements"
            ) as mock_measurement_scorer,
        ):
            # Setup mocks
            mock_topology_scorer.return_value = MagicMock(return_value=topology_score)
            mock_measurement_scorer.return_value = MagicMock(
                return_value=measurement_score
            )

            expected_topology = {
                "resistors": {"R1": 10.0},
                "connections": [["A", "B", "R1"]],
            }
            expected_measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

            scorer = check_complete_circuit_solution(
                expected_topology,
                expected_measurements,
                topology_weight=topology_weight,
                measurement_weight=measurement_weight,
            )

            topology_input = (
                '{"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}'
            )
            final_score = scorer(topology_input)

            expected_final_score = (
                topology_weight * topology_score
                + measurement_weight * measurement_score
            )
            assert abs(final_score - expected_final_score) < 1e-6

    @given(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
    def test_tolerance_bounds_property(self, topo_weight, meas_weight, tolerance):
        """Property: Function should accept any valid tolerance value"""
        with (
            patch(
                "resistor_network.score.check_resistor_topology"
            ) as mock_topology_scorer,
            patch(
                "resistor_network.score.check_resistance_measurements"
            ) as mock_measurement_scorer,
        ):
            mock_topology_scorer.return_value = MagicMock(return_value=0.5)
            mock_measurement_scorer.return_value = MagicMock(return_value=0.5)

            expected_topology = {
                "resistors": {"R1": 10.0},
                "connections": [["A", "B", "R1"]],
            }
            expected_measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

            # Should not raise exception with any valid tolerance
            scorer = check_complete_circuit_solution(
                expected_topology,
                expected_measurements,
                topology_weight=topo_weight,
                measurement_weight=meas_weight,
                tolerance=tolerance,
            )

            assert callable(scorer)

            # Verify tolerance was passed to both scorers
            mock_topology_scorer.assert_called_once_with(
                expected_topology=expected_topology,
                use_functional_scoring=False,
                topology_weight=0.5,
                functional_weight=0.0,
                exact_values_weight=0.5,
                tolerance=tolerance,
                require_both=True,
            )
            mock_measurement_scorer.assert_called_once_with(
                expected_measurements, tolerance
            )

    @given(score_pairs())
    def test_score_bounds_property(self, scores):
        """Property: Final score should always be in valid range [0.0, 1.0]"""
        topology_score, measurement_score = scores

        with (
            patch(
                "resistor_network.score.check_resistor_topology"
            ) as mock_topology_scorer,
            patch(
                "resistor_network.score.check_resistance_measurements"
            ) as mock_measurement_scorer,
        ):
            # Setup mocks
            mock_topology_scorer.return_value = MagicMock(return_value=topology_score)
            mock_measurement_scorer.return_value = MagicMock(
                return_value=measurement_score
            )

        expected_topology = {
            "resistors": {"R1": 10.0},
            "connections": [["A", "B", "R1"]],
        }
        expected_measurements = [{"node_a": "A", "node_b": "B", "resistance": 10.0}]

        # Use default weights that sum to 1.0
        scorer = check_complete_circuit_solution(
            expected_topology, expected_measurements
        )

        topology_input = (
            '{"resistors": {"R1": 10.0}, "connections": [["A", "B", "R1"]]}'
        )
        final_score = scorer(topology_input)

        # Since weights sum to 1.0 and individual scores are [0,1], final score must be [0,1]
        assert 0.0 <= final_score <= 1.0
