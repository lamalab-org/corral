import h5py
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_hdf5_file_exists():
    """Test that the HDF5 file exists."""
    data_path = Path(__file__).parent.parent / "data" / "experimental_data.h5"
    assert data_path.exists(), f"HDF5 file not found at {data_path}"
    print("✓ HDF5 file exists")


def test_hdf5_structure():
    """Test the structure of the HDF5 file."""
    data_path = Path(__file__).parent.parent / "data" / "experimental_data.h5"
    
    with h5py.File(data_path, 'r') as f:
        # Check that there are experiments
        exp_names = list(f.keys())
        assert len(exp_names) > 0, "No experiments found in HDF5 file"
        print(f"✓ Found {len(exp_names)} experiments")
        
        # Check first experiment structure
        first_exp = exp_names[0]
        exp_group = f[first_exp]
        
        # Check for required groups
        assert 'datasets' in exp_group, f"Missing 'datasets' group in {first_exp}"
        assert 'experiment_metadata' in exp_group, f"Missing 'experiment_metadata' group in {first_exp}"
        print(f"✓ Required groups present in {first_exp}")
        
        # Check data_corrected structure
        assert 'data_corrected' in exp_group['datasets'], \
            f"Missing 'data_corrected' dataset in {first_exp}/datasets"
        
        data_corrected = exp_group['datasets']['data_corrected']
        assert data_corrected.ndim == 2, \
            f"data_corrected should be 2D, got {data_corrected.ndim}D"
        assert data_corrected.shape[1] == 2, \
            f"data_corrected should have 2 columns, got {data_corrected.shape[1]}"
        assert data_corrected.shape[0] > 0, \
            f"data_corrected should have rows, got {data_corrected.shape[0]}"
        print(f"✓ data_corrected has correct structure: {data_corrected.shape}")
        
        # Check metadata structure
        meta_group = exp_group['experiment_metadata']
        expected_fields = [
            'ru_concentration',
            'oxidant_concentration',
            'pH',
            'power_output'
        ]
        
        for field in expected_fields:
            assert field in meta_group, \
                f"Missing expected metadata field '{field}' in {first_exp}"
        print(f"✓ All expected metadata fields present")
        
        # Check metadata types
        assert isinstance(meta_group['ru_concentration'][()], (float, np.floating)), \
            "ru_concentration should be numeric"
        assert isinstance(meta_group['oxidant_concentration'][()], (float, np.floating)), \
            "oxidant_concentration should be numeric"
        assert isinstance(meta_group['pH'][()], (float, np.floating)), \
            "pH should be numeric"
        print(f"✓ Metadata fields have correct types")


def test_data_values():
    """Test that data values are reasonable."""
    data_path = Path(__file__).parent.parent / "data" / "experimental_data.h5"
    
    with h5py.File(data_path, 'r') as f:
        first_exp = list(f.keys())[0]
        
        # Check time values
        data_corrected = np.array(f[first_exp]['datasets']['data_corrected'])
        time = data_corrected[:, 0]
        oxygen = data_corrected[:, 1]
        
        assert time[0] >= 0, "Time should start at or after 0"
        assert np.all(np.diff(time) >= 0), "Time should be monotonically increasing"
        assert time[-1] > 0, "Experiment should have non-zero duration"
        print(f"✓ Time data is valid: range [{time[0]:.1f}, {time[-1]:.1f}] s")
        
        # Check oxygen values
        assert np.all(oxygen >= 0), "Oxygen concentration should be non-negative"
        assert np.all(np.isfinite(oxygen)), "Oxygen values should be finite"
        print(f"✓ Oxygen data is valid: range [{oxygen.min():.3f}, {oxygen.max():.3f}] µM")
        
        # Check metadata values
        meta = f[first_exp]['experiment_metadata']
        
        ru_conc = meta['ru_concentration'][()]
        assert 0 < ru_conc < 1000, f"Ru concentration seems unreasonable: {ru_conc} µM"
        print(f"✓ Ru concentration is reasonable: {ru_conc} µM")
        
        oxidant_conc = meta['oxidant_concentration'][()]
        assert 0 < oxidant_conc < 50000, \
            f"Oxidant concentration seems unreasonable: {oxidant_conc} µM"
        print(f"✓ Oxidant concentration is reasonable: {oxidant_conc} µM")
        
        pH = meta['pH'][()]
        assert 0 < pH < 14, f"pH value seems unreasonable: {pH}"
        print(f"✓ pH is reasonable: {pH}")


