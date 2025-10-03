#!/bin/bash

# Kinetic Fitting Debug Runner
# This script helps you run different debugging tools to identify flat line issues

echo "=================================================="
echo "    KINETIC FITTING DEBUG TOOLS"
echo "=================================================="
echo ""
echo "Available debug options:"
echo "  1) Quick debug test (single experiment)"
echo "  2) Full debugging sequence" 
echo "  3) Parameter sweep analysis"
echo "  4) Run all debugging tools"
echo ""

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "❌ Error: data/ directory not found"
    echo "   Make sure you're running this from the kinetic_fitting directory"
    echo "   and that experimental_data.h5 exists in data/"
    exit 1
fi

# Function to run a debug script
run_debug_script() {
    local script_name=$1
    local description=$2
    
    echo ""
    echo "🔧 $description"
    echo "Running: python3 $script_name"
    echo "----------------------------------------"
    
    if python3 "$script_name"; then
        echo "✅ $description completed successfully"
    else
        echo "❌ $description failed"
    fi
    
    echo ""
}

# Get user choice
if [ $# -eq 0 ]; then
    echo -n "Enter your choice (1-4): "
    read choice
else
    choice=$1
fi

case $choice in
    1)
        echo "Selected: Quick debug test"
        run_debug_script "quick_debug.py" "Quick ODE system test with reference parameters"
        ;;
    2)
        echo "Selected: Full debugging sequence"
        run_debug_script "debug_fitting.py" "Comprehensive debugging analysis"
        ;;
    3)
        echo "Selected: Parameter sweep analysis"
        run_debug_script "parameter_sweep.py" "Parameter sweep to identify good vs bad ranges"
        ;;
    4)
        echo "Selected: Run all debugging tools"
        run_debug_script "quick_debug.py" "Quick ODE system test"
        run_debug_script "parameter_sweep.py" "Parameter sweep analysis"
        run_debug_script "debug_fitting.py" "Full debugging sequence"
        ;;
    *)
        echo "❌ Invalid choice. Please select 1-4."
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "    DEBUG SESSION COMPLETE"
echo "=================================================="
echo ""
echo "📁 Check these locations for output files:"
echo "   - Plots and detailed logs in persistent output directory"
echo "   - ode_test_*.png - ODE system test plots"
echo "   - reference_fit_*.png - Reference fit plots"  
echo "   - fit_params_*.json - Detailed parameter logs"
echo ""
echo "💡 Next steps based on results:"
echo "   - If ODE tests show flat lines: Issue with ODE system/network"
echo "   - If ODE works but optimization fails: Issue with bounds/penalties"
echo "   - If parameter sweep shows no good sets: Fundamental network problem"