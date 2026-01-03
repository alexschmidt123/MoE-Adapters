#!/bin/bash
# Script to download all required datasets for MTIL training

set -e

# Configuration: Set your data location here
# Default: use relative path to datasets directory (one level up from mtil)
DATA_LOCATION="${DATA_LOCATION:-$(cd "$(dirname "$0")/../.." && pwd)/datasets}"

echo "=========================================="
echo "Downloading all required datasets"
echo "=========================================="
echo "Data location: ${DATA_LOCATION}"
echo ""

# Ensure we're in the mtil directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Run the Python download script
python download_all_datasets.py

echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo ""
echo "Note: If Caltech101 failed, you may need to download it manually."
echo "See DOWNLOAD_CALTECH101.md for instructions."
