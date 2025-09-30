"""Comprehensive smoke test for kinetic fitting environment."""

import sys
import tempfile
from pathlib import Path

from kinetic_fitting.tools import (
    load_experimental_data,
    describe_experimental_data,
    get_current_network,
    fit_single_experiment,
    initialize_default_network,
    setup_working_directory,
    save_reaction_network,
)

def main():
    print("=" * 60)
    print("KINETIC FITTING ENVIRONMENT SMOKE TEST")
    print("=" * 60)
    
    # Setup test environment
    with tempfile.TemporaryDirectory(prefix="kinetic_test_") as temp_dir:
        work_dir = Path(temp_dir)
        data_path = Path(__file__).parent / "data" / "experimental_data.h5"
        network_path = work_dir / "current_network.json"
        results_path = work_dir / "fit_results.json"
        
        print(f"Test directory: {temp_dir}")
        print(f"Data path: {data_path}")
        
        # Test 1: Check data file exists
        print("\n" + "="*40)
        print("TEST 1: Data file exists")
        print("="*40)
        if not data_path.exists():
            print(f"❌ FAIL: Data file not found at {data_path}")
            return False
        print(f"✅ PASS: Data file found")
        
        # Test 2: Load experimental data
        print("\n" + "="*40)
        print("TEST 2: Load experimental data")
        print("="*40)
        try:
            data = load_experimental_data(str(data_path))
            print(f"✅ PASS: Loaded {len(data)} experiments")
            
            # Check first experiment structure
            first_exp = list(data.keys())[0]
            exp_data = data[first_exp]
            required_keys = ['time', 'oxygen', 'metadata']
            for key in required_keys:
                if key not in exp_data:
                    print(f"❌ FAIL: Missing key '{key}' in experiment data")
                    return False
            print(f"✅ PASS: Experiment data has correct structure")
            
            # Check metadata
            meta = exp_data['metadata']
            if meta.get('c_Ru') is None:
                print(f"❌ FAIL: Missing Ru concentration in metadata")
                return False
            print(f"✅ PASS: Metadata has Ru concentration: {meta['c_Ru']} µM")
            
        except Exception as e:
            print(f"❌ FAIL: Error loading data: {e}")
            return False
        
        # Test 3: Setup working directory and create default network
        print("\n" + "="*40)
        print("TEST 3: Setup working directory")
        print("="*40)
        try:
            setup_working_directory(str(work_dir), str(data_path))
            
            if not network_path.exists():
                print(f"❌ FAIL: Network file not created")
                return False
            if not results_path.exists():
                print(f"❌ FAIL: Results file not created")
                return False
            
            print(f"✅ PASS: Working directory setup complete")
        except Exception as e:
            print(f"❌ FAIL: Error setting up working directory: {e}")
            return False
        
        # Test 4: Test describe_experimental_data tool
        print("\n" + "="*40)
        print("TEST 4: Describe experimental data")
        print("="*40)
        try:
            result = describe_experimental_data(str(data_path))
            if "µM" not in result or "experiments" not in result.lower():
                print(f"❌ FAIL: Description doesn't contain expected content")
                return False
            print(f"✅ PASS: Data description generated ({len(result)} chars)")
            print(result[:200] + "..." if len(result) > 200 else result)
        except Exception as e:
            print(f"❌ FAIL: Error describing data: {e}")
            return False
        
        # Test 5: Test get_current_network tool
        print("\n" + "="*40)
        print("TEST 5: Get current network")
        print("="*40)
        try:
            result = get_current_network(str(network_path))
            if "RuII" not in result or "reactions" not in result.lower():
                print(f"❌ FAIL: Network description doesn't contain expected content")
                return False
            print(f"✅ PASS: Network description generated ({len(result)} chars)")
            print(result[:300] + "..." if len(result) > 300 else result)
        except Exception as e:
            print(f"❌ FAIL: Error getting network: {e}")
            return False
        
        # Test 6: Test single experiment fitting
        print("\n" + "="*40)
        print("TEST 6: Fit single experiment")
        print("="*40)
        try:
            exp_name = list(data.keys())[0]
            print(f"Fitting experiment: {exp_name}")
            
            import time
            start = time.time()
            result = fit_single_experiment(
                str(data_path),
                str(network_path), 
                str(results_path),
                exp_name
            )
            elapsed = time.time() - start
            
            if "R²" not in result:
                print(f"❌ FAIL: Fit result doesn't contain R²")
                return False
            if elapsed > 30:
                print(f"❌ FAIL: Fit took too long: {elapsed:.1f}s")
                return False
            
            print(f"✅ PASS: Fit completed in {elapsed:.1f}s")
            print(result)
        except Exception as e:
            print(f"❌ FAIL: Error fitting experiment: {e}")
            return False
        
        # Test 7: Test stoichiometry parsing for complex reactions
        print("\n" + "="*40)
        print("TEST 7: Stoichiometry parsing")
        print("="*40)
        try:
            from kinetic_fitting.tools import Reaction
            
            # Test simple reaction
            rxn1 = Reaction.from_dict({
                "equation": "A + B -> C",
                "type": "dark",
                "k_range": [1e3, 1e5]
            })
            expected1 = {"A": -1, "B": -1, "C": 1}
            if rxn1.stoichiometry != expected1:
                print(f"❌ FAIL: Wrong stoichiometry for simple reaction: {rxn1.stoichiometry}")
                return False
            
            # Test complex reaction with coefficients  
            rxn2 = Reaction.from_dict({
                "equation": "2 RuIII + H2O2 -> 2 RuII + O2 + 2 H",
                "type": "dark", 
                "k_range": [1e3, 1e5]
            })
            expected2 = {"RuIII": -2, "H2O2": -1, "RuII": 2, "O2": 1}
            if rxn2.stoichiometry != expected2:
                print(f"❌ FAIL: Wrong stoichiometry for complex reaction: {rxn2.stoichiometry}")
                print(f"Expected: {expected2}")
                print(f"Got: {rxn2.stoichiometry}")
                return False
            
            print(f"✅ PASS: Stoichiometry parsing works correctly")
        except Exception as e:
            print(f"❌ FAIL: Error in stoichiometry parsing: {e}")
            return False
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED - System is working!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)