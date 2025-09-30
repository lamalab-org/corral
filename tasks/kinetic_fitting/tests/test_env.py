from ..env import KineticFittingEnv
import numpy as np


def test_environment_loading():
    """Test that the environment loads correctly."""

    env = KineticFittingEnv()
    
    # Check that data was loaded
    assert len(env.experimental_data) > 0, "No experiments loaded"
    print(f"✓ Environment loaded {len(env.experimental_data)} experiments")
    
    # Check first experiment
    first_exp_name = list(env.experimental_data.keys())[0]
    first_exp = env.experimental_data[first_exp_name]
    
    # Check data structure
    assert 'time' in first_exp, "Missing 'time' in experiment data"
    assert 'oxygen' in first_exp, "Missing 'oxygen' in experiment data"
    assert 'metadata' in first_exp, "Missing 'metadata' in experiment data"
    print(f"✓ Experiment data has correct structure")
    
    # Check data types
    assert isinstance(first_exp['time'], np.ndarray), "time should be numpy array"
    assert isinstance(first_exp['oxygen'], np.ndarray), "oxygen should be numpy array"
    assert isinstance(first_exp['metadata'], dict), "metadata should be dict"
    print(f"✓ Data types are correct")
    
    # Check array shapes match
    assert first_exp['time'].shape == first_exp['oxygen'].shape, \
        "time and oxygen arrays should have same shape"
    print(f"✓ Time and oxygen arrays have matching shapes: {first_exp['time'].shape}")
    
    # Check metadata mapping
    meta = first_exp['metadata']
    required_fields = ['c_Ru', 'c_S2O8', 'pH', 'irradiance']
    for field in required_fields:
        assert field in meta, f"Missing required field '{field}' in metadata"
        assert meta[field] is not None, f"Field '{field}' should not be None"
    print(f"✓ All required metadata fields present and non-null")
    
    # Check metadata values are reasonable
    assert 0 < meta['c_Ru'] < 1000, f"c_Ru unreasonable: {meta['c_Ru']}"
    assert 0 < meta['c_S2O8'] < 50000, f"c_S2O8 unreasonable: {meta['c_S2O8']}"
    assert 0 < meta['pH'] < 14, f"pH unreasonable: {meta['pH']}"
    print(f"✓ Metadata values are reasonable")
    print(f"  - c_Ru: {meta['c_Ru']} µM")
    print(f"  - c_S2O8: {meta['c_S2O8']} µM")
    print(f"  - pH: {meta['pH']}")
    print(f"  - irradiance/power: {meta['irradiance']}")


def test_reaction_network():
    """Test that the initial reaction network is valid."""
    env = KineticFittingEnv()
    network = env.current_reaction_network
    
    assert 'reactions' in network, "Network should have 'reactions' key"
    assert len(network['reactions']) > 0, "Network should have at least one reaction"
    print(f"✓ Reaction network has {len(network['reactions'])} reactions")
    
    # Check each reaction has required fields
    for i, rxn in enumerate(network['reactions']):
        assert 'equation' in rxn, f"Reaction {i} missing 'equation'"
        assert 'type' in rxn, f"Reaction {i} missing 'type'"
        assert rxn['type'] in ['light', 'dark'], \
            f"Reaction {i} has invalid type: {rxn['type']}"
        
        if rxn['type'] == 'light':
            assert 'quantum_yield' in rxn, \
                f"Light reaction {i} missing 'quantum_yield'"
            qy = rxn['quantum_yield']
            assert 0 <= qy[0] <= 1 and 0 <= qy[1] <= 1, \
                f"Quantum yield out of range [0,1]: {qy}"
        else:
            assert 'k_range' in rxn, f"Dark reaction {i} missing 'k_range'"
            k_range = rxn['k_range']
            assert k_range[0] > 0 and k_range[1] > k_range[0], \
                f"Invalid k_range: {k_range}"
    
    print(f"✓ All reactions have valid structure")


def test_environment_state():
    """Test environment state methods."""
    
    env = KineticFittingEnv()
    
    # Test get_state
    state = env.get_state()
    assert isinstance(state, dict), "State should be a dict"
    assert 'num_experiments' in state, "State should have 'num_experiments'"
    assert 'best_score' in state, "State should have 'best_score'"
    assert 'num_iterations' in state, "State should have 'num_iterations'"
    print(f"✓ get_state() returns correct structure")
    
    # Test reset
    env.best_score = 0.5
    env.fit_history = [1, 2, 3]
    env.reset()
    assert env.best_score == 0.0, "reset() should reset best_score"
    assert len(env.fit_history) == 0, "reset() should clear fit_history"
    print(f"✓ reset() works correctly")

