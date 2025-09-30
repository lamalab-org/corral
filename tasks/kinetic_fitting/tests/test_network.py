"""Test network JSON creation and loading."""

import json
import tempfile
from pathlib import Path

import pytest


def test_network_json_creation_and_loading():
    """Test network JSON creation and loading."""
    network = {
        'reactions': [
            {'equation': 'RuII + hv -> RuII*', 'type': 'light', 'quantum_yield': [0.8, 1.0]},
            {'equation': 'RuII* + S2O8 -> RuIII + SO4_rad + SO4', 'type': 'dark', 'k_range': [1e7, 1e9]},
            {'equation': '2 RuIII + H2O2 -> 2 RuII + O2 + 2 H', 'type': 'dark', 'k_range': [1e3, 1e5]},
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(network, f, indent=2)
        temp_path = f.name
    
    try:
        # Read it back
        with open(temp_path) as f:
            loaded_network = json.load(f)
        
        assert len(loaded_network['reactions']) == 3
        assert loaded_network['reactions'][0]['equation'] == 'RuII + hv -> RuII*'
        assert loaded_network['reactions'][0]['type'] == 'light'
        assert loaded_network['reactions'][0]['quantum_yield'] == [0.8, 1.0]
        
        assert loaded_network['reactions'][1]['equation'] == 'RuII* + S2O8 -> RuIII + SO4_rad + SO4'
        assert loaded_network['reactions'][1]['type'] == 'dark'
        assert loaded_network['reactions'][1]['k_range'] == [1e7, 1e9]
        
        assert loaded_network['reactions'][2]['equation'] == '2 RuIII + H2O2 -> 2 RuII + O2 + 2 H'
        assert loaded_network['reactions'][2]['k_range'] == [1e3, 1e5]
        
    finally:
        # Clean up
        Path(temp_path).unlink()


def test_empty_network():
    """Test handling of empty network."""
    network = {'reactions': []}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(network, f, indent=2)
        temp_path = f.name
    
    try:
        with open(temp_path) as f:
            loaded_network = json.load(f)
        
        assert len(loaded_network['reactions']) == 0
        assert loaded_network['reactions'] == []
        
    finally:
        Path(temp_path).unlink()


def test_network_with_metadata():
    """Test network with additional metadata."""
    network = {
        'reactions': [
            {'equation': 'A -> B', 'type': 'dark', 'k_range': [1e3, 1e5]}
        ],
        'metadata': {
            'created_by': 'test',
            'version': '1.0',
            'description': 'Test network'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(network, f, indent=2)
        temp_path = f.name
    
    try:
        with open(temp_path) as f:
            loaded_network = json.load(f)
        
        assert 'metadata' in loaded_network
        assert loaded_network['metadata']['created_by'] == 'test'
        assert loaded_network['metadata']['version'] == '1.0'
        assert loaded_network['metadata']['description'] == 'Test network'
        
    finally:
        Path(temp_path).unlink()


def test_invalid_json_handling():
    """Test handling of invalid JSON."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('invalid json content {')
        temp_path = f.name
    
    try:
        with pytest.raises(json.JSONDecodeError):
            with open(temp_path) as f:
                json.load(f)
    finally:
        Path(temp_path).unlink()