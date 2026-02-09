#!/bin/bash
# Synapse Benchmark Quick Start Script

set -e  # Exit on error

echo "======================================"
echo "Synapse Benchmark Quick Start Setup"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if ! python -c "import sys; assert sys.version_info >= (3, 7) and sys.version_info < (3, 9)" 2>/dev/null; then
    echo "Error: Python 3.7 or 3.8 is required"
    exit 1
fi

# Create directories
echo ""
echo "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p checkpoints
mkdir -p results
mkdir -p outputs
mkdir -p logs

echo "✓ Directories created"

# Install dependencies
echo ""
echo "Installing dependencies..."
read -p "Do you want to install dependencies now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements.txt
    pip install -e .
    echo "✓ Dependencies installed"
else
    echo "Skipping dependency installation"
    echo "You can install later with: pip install -r requirements.txt && pip install -e ."
fi

# Check for CUDA
echo ""
echo "Checking CUDA availability..."
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "✓ CUDA $cuda_version found"
else
    echo "⚠ CUDA not found. GPU acceleration will not be available."
fi

# Data setup instructions
echo ""
echo "======================================"
echo "Next Steps"
echo "======================================"
echo ""
echo "1. Download the required data files:"
echo "   See data/README.md for download instructions"
echo "   Place .npy files in: data/raw/"
echo ""
echo "2. (Optional) Download pre-trained checkpoints:"
echo "   See checkpoints/README.md for download instructions"
echo "   Place checkpoints in: checkpoints/{ENV_NAME}/"
echo ""
echo "3. Configure your experiment:"
echo "   Edit configs/config_template.yaml or create a new config"
echo ""
echo "4. Run a quick test:"
echo "   python scripts/train_policy.py --config configs/RZCH_clinical.yaml --max_episodes 10"
echo ""
echo "For more information, see README.md"
echo ""
echo "Setup complete! 🎉"
