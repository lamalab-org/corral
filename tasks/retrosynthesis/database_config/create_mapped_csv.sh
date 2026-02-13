#!/bin/bash

# Exit on any error
set -e

echo "Step 1: Creating virtual environment with uv (Python 3.12)..."
uv venv --python 3.12

echo "Step 1.5: Activating virtual environment..."
source .venv/bin/activate

echo "Step 2: Installing Python packages..."
uv pip install reaction-utils metaflow ord-schema setuptools

echo "Step 2.5: Cloning ord-data repository..."
git clone https://github.com/open-reaction-database/ord-data.git

echo "Step 3: Running ORD preparation pipeline..."
python -m rxnutils.data.ord.preparation_pipeline run --nbatches 200 --max-workers 8 --max-num-splits 200 --ord-data ord-data

echo "Step 4: Removing ord-data directory..."
rm -rf ord-data/

echo "Step 5: Running USPTO preparation pipeline..."
python -m rxnutils.data.uspto.preparation_pipeline run --nbatches 200 --max-workers 8 --max-num-splits 200

echo "Step 5.5-5.8: Cleaning up USPTO files..."
rm -f 2001_Sep2016_USPTOapplications_smiles.rsmi.7z
rm -f 2001_Sep2016_USPTOapplications_smiles.rsmi
rm -f 1976_Sep2016_USPTOgrants_smiles.rsmi.7z
rm -f 1976_Sep2016_USPTOgrants_smiles.rsmi

echo "Step 6: Installing rxnmapper..."
uv pip install rxnmapper

echo "Step 7: Running mapping pipeline for ORD data..."
python -m rxnutils.data.mapping_pipeline run --data-prefix ord --nbatches 200 --max-workers 8 --max-num-splits 200

echo "Step 8: Running mapping pipeline for USPTO data..."
python -m rxnutils.data.mapping_pipeline run --data-prefix uspto --nbatches 200 --max-workers 8 --max-num-splits 200

rm -f ord_data_cleaned.csv ord_data.csv
rm -f uspto_data_cleaned.csv uspto_data.csv

echo "All steps completed successfully!"
