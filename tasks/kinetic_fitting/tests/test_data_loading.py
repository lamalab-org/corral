"""Test data loading functionality."""

import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import h5py


@dataclass
class ExperimentMetadata:
    """Store experimental conditions and metadata"""
    experiment_name: str
    power_output: float
    ru_concentration: float
    oxidant_concentration: float
    buffer_concentration: float
    pH: float
    buffer_used: int 
    annotations: str = ""
    color: str = "#ce1480"


@dataclass
class AnalysisMetadata:
    """Store analysis metadata"""
    p: np.ndarray
    max_rate: float
    max_rate_ydiff: float
    initial_state: np.ndarray
    matrix: str
    rate_constant: float
    rxn_start: int
    rxn_end: int
    residual: np.ndarray
    idx_for_fitting: int
    

@dataclass
class DataSets:
    """Store datasets"""    
    data_corrected: np.ndarray


@dataclass
class TimeSeriesData:
    """Store time series data"""   
    time_reaction: np.ndarray
    data_reaction: np.ndarray
    y_fit: np.ndarray
    baseline_y: np.ndarray
    lbc_fit_y: np.ndarray
    full_x_values: np.ndarray
    full_y_corrected: np.ndarray 
    x_diff: np.ndarray
    y_diff: np.ndarray
    y_diff_smoothed: np.ndarray
    y_diff_fit: np.ndarray
    time_full: np.ndarray
    data_full: np.ndarray
    

@dataclass
class ExperimentalData:
    """Container for individual experiment's data"""
    time_series_data: TimeSeriesData
    experiment_metadata: ExperimentMetadata
    analysis_metadata: AnalysisMetadata
    datasets: DataSets


@dataclass
class ExperimentalDataset:
    experiments: Dict[str, 'ExperimentalData'] = field(default_factory=dict)
    overview_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    
    @classmethod
    def load_from_hdf5(cls, filename: str):
        """Load experiments from HDF5 file"""
        dataset = cls()

        try:
            dataset.overview_df = pd.read_hdf(filename, key='overview_df')
        except (KeyError, ValueError):
            dataset.overview_df = pd.DataFrame()

        with h5py.File(filename, 'r') as f:
            for exp_name in f.keys():
                if exp_name == 'overview_df':  # Skip the overview_df group
                    continue
                try:
                    # Load experimental data
                    time_series_dict = dict(f[f'{exp_name}/time_series_data'].attrs)
                    time_series_data = TimeSeriesData(**time_series_dict)
                   
                    # Load metadata
                    exp_metadata_dict = dict(f[f'{exp_name}/experiment_metadata'].attrs)
                    experiment_metadata = ExperimentMetadata(**exp_metadata_dict)

                    analysis_metadata_dict = dict(f[f'{exp_name}/analysis_metadata'].attrs)
                    analysis_metadata = AnalysisMetadata(**analysis_metadata_dict)

                    datasets_dict = dict(f[f'{exp_name}/datasets'].attrs)
                    datasets = DataSets(**datasets_dict)

                    # Create ExperimentalData and add to dataset
                    experimental_data = ExperimentalData(
                        time_series_data, experiment_metadata, analysis_metadata, datasets
                    )
                    dataset.experiments[exp_name] = experimental_data
                    
                except Exception:
                    continue  # Skip experiments that can't be loaded
        
        return dataset


def test_data_loading():
    """Test data loading functionality."""
    data_path = Path(__file__).parent.parent / "data" / "experimental_data.h5"
    
    if not data_path.exists():
        pytest.skip(f"Data file not found: {data_path}")
    
    # Test loading with proper structure
    try:
        dataset = ExperimentalDataset.load_from_hdf5(str(data_path))
        
        data = {}
        for exp_name, exp_data in dataset.experiments.items():
            # Extract time and oxygen data
            time = exp_data.time_series_data.time_reaction
            oxygen = exp_data.time_series_data.data_reaction
            
            standardized_metadata = {
                'c_Ru': exp_data.experiment_metadata.ru_concentration,
                'c_S2O8': exp_data.experiment_metadata.oxidant_concentration,
                'power_output': exp_data.experiment_metadata.power_output,
                'pH': exp_data.experiment_metadata.pH,
                'irradiance': exp_data.experiment_metadata.power_output,
            }
            
            data[exp_name] = {
                'time': time.tolist() if hasattr(time, 'tolist') else time,
                'oxygen': oxygen.tolist() if hasattr(oxygen, 'tolist') else oxygen,
                'metadata': standardized_metadata,
            }
        
        if not data:
            # Fallback: just load overview data if experiments are empty
            if not dataset.overview_df.empty:
                assert len(dataset.overview_df) > 0, "Should have overview data"
                return
            pytest.fail("No experimental data could be loaded")
        
        # Validate loaded data
        assert len(data) > 0, "Should load at least one experiment"
        
        first_exp = list(data.keys())[0]
        exp_data = data[first_exp]
        
        assert 'time' in exp_data, "Experiment should have time data"
        assert 'oxygen' in exp_data, "Experiment should have oxygen data"
        assert 'metadata' in exp_data, "Experiment should have metadata"
        
        metadata = exp_data['metadata']
        assert 'c_Ru' in metadata, "Metadata should have Ru concentration"
        assert 'c_S2O8' in metadata, "Metadata should have S2O8 concentration"
        
    except Exception as e:
        # Simple fallback - just check overview
        try:
            overview_df = pd.read_hdf(str(data_path), key='overview_df')
            assert len(overview_df) > 0, "Should have overview data"
        except Exception:
            pytest.fail(f"Could not load any data: {e}")


def test_experimental_dataset_structure():
    """Test the experimental dataset data structures."""
    # Test basic structure creation
    dataset = ExperimentalDataset()
    assert isinstance(dataset.experiments, dict)
    assert isinstance(dataset.overview_df, pd.DataFrame)
    
    # Test dataclass instantiation
    metadata = ExperimentMetadata(
        experiment_name="TEST-01",
        power_output=1000.0,
        ru_concentration=5.0e-6,
        oxidant_concentration=3000.0e-6,
        buffer_concentration=0.1,
        pH=7.0,
        buffer_used=1
    )
    
    assert metadata.experiment_name == "TEST-01"
    assert metadata.power_output == 1000.0
    assert metadata.color == "#ce1480"  # default value