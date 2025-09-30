"""Simple test to verify data loading works."""

import sys
from pathlib import Path
def test_data_exists():
    """Test that the HDF5 file exists."""
    data_path = Path(__file__).parent / "data" / "experimental_data.h5"
    print(f"Checking for data file at: {data_path}")
    
    if data_path.exists():
        print(f"✅ Data file found")
        print(f"   Size: {data_path.stat().st_size / (1024*1024):.1f} MB")
        return True
    else:
        print(f"❌ Data file not found")
        return False

def test_can_import_tools():
    """Test that we can import the tools."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from kinetic_fitting.tools import load_experimental_data
        print("✅ Successfully imported tools")
        return True
    except Exception as e:
        print(f"❌ Failed to import tools: {e}")
        return False

def test_load_data():
    """Test loading the experimental data."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from kinetic_fitting.tools import load_experimental_data
        
        data_path = Path(__file__).parent / "data" / "experimental_data.h5"
        data = load_experimental_data(str(data_path))
        
        print(f"✅ Loaded {len(data)} experiments")
        
        # Check first experiment
        if len(data) > 0:
            first_exp = list(data.keys())[0]
            exp_data = data[first_exp]
            print(f"   First experiment: {first_exp}")
            print(f"   Keys: {list(exp_data.keys())}")
            print(f"   Time points: {len(exp_data['time'])}")
            print(f"   Metadata keys: {list(exp_data['metadata'].keys())}")
            
        return True
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SIMPLE DATA LOADING TEST")
    print("=" * 50)
    
    success = True
    success &= test_data_exists()
    success &= test_can_import_tools()
    success &= test_load_data()
    
    if success:
        print("\n🎉 All basic tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    sys.exit(0 if success else 1)