#!/bin/bash
# Convenience script for running EZKL CNN benchmarks

set -e

echo "EZKL CNN Benchmark Runner"
echo "=========================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if ! python -c "import ezkl" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Parse command line arguments
case "${1:-all}" in
    all)
        echo "Running all experiments..."
        python src/run_experiments.py
        ;;
    core)
        echo "Running core layer experiments..."
        python src/run_experiments.py --core-only
        ;;
    scaling)
        echo "Running scaling experiments..."
        python src/run_experiments.py --scaling-only
        ;;
    composite)
        echo "Running composite architecture experiments..."
        shift
        python src/run_experiments.py --composite-only "$@"
        ;;
    analyze)
        echo "Analyzing results..."
        python src/analyze_results.py
        ;;
    train)
        echo "Training composite models..."
        python src/train_cifar10.py
        ;;
    test)
        echo "Testing setup..."
        python -c "
import torch
import ezkl
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'EZKL version: {ezkl.__version__}')
print('Setup OK!')
        "
        ;;
    clean)
        echo "Cleaning temporary files..."
        rm -rf temp/*
        echo "Cleaned temp directory"
        ;;
    help|*)
        echo "Usage: ./run_benchmark.sh [command]"
        echo ""
        echo "Commands:"
        echo "  all          Run all 40 experiments (default)"
        echo "  core         Run only core layer experiments (24)"
        echo "  scaling      Run only scaling study (8)"
        echo "  composite    Run only composite experiments (8)"
        echo "  analyze      Analyze results and generate tables"
        echo "  train        Train composite models only"
        echo "  test         Test environment setup"
        echo "  clean        Clean temporary files"
        echo "  help         Show this help message"
        ;;
esac
