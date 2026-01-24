#!/bin/bash

# Trace Analysis - Quick Setup Script
# ====================================

echo "=========================================="
echo "Trace Success Prediction - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python --version

if [ $? -ne 0 ]; then
    echo "❌ Python not found! Please install Python 3.7+"
    exit 1
fi

echo "✓ Python found"
echo ""

# Install dependencies
echo "Installing required packages..."
echo "This may take a few minutes..."
echo ""

pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Installation failed!"
    echo "Try installing manually: pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels xgboost shap"
    exit 1
fi

echo ""
echo "✓ All packages installed successfully!"
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit trace_analysis.py and update CONFIG['input_file'] with your CSV path"
echo "2. Run: python trace_analysis.py"
echo ""
echo "See README.md for detailed instructions"
echo ""