# EZKL CNN Benchmark

Automated benchmarking framework for CNN components in EZKL, supporting the paper:

**"Benchmarking CNN Components for Verifiable Inference on EVM-Compatible Blockchains: A Layer-Level Analysis of EZKL Performance"**

> **Note:** The paper manuscript is not included in this repository and will be made available upon publication. This repository contains all experimental code, results, and analysis scripts to reproduce the findings.

## Overview

This repository provides a comprehensive benchmarking pipeline that:
- Profiles **8 feasible CNN layer types** across 2 tolerance settings (16 experiments)
- Conducts a **scaling study** on Dense layers (4 experiments)
- Evaluates **3 working composite CNN architectures** trained on CIFAR-10 (6 experiments)
- Measures proof-generation time, proof size, circuit complexity, and peak memory
- Generates **26 feasible experiments** on 125GB RAM hardware
- Documents **10 additional experiments** excluded due to memory constraints (Conv2d, LayerNorm, CNN_ReLU)

## Project Structure

```
cnn_ezkl_bench/
├── config/
│   └── experiment_config.py    # Experiment configurations
├── src/
│   ├── models.py                # PyTorch model definitions
│   ├── train_cifar10.py         # CIFAR-10 training utilities
│   ├── ezkl_utils.py            # EZKL integration and benchmarking
│   ├── run_experiments.py       # Automated experiment runner
│   └── analyze_results.py       # Results analysis and visualization
├── analysis/
│   └── results_summary.csv      # Summary of all experiments
├── results/                     # Benchmark results (JSON)
│   ├── core_layers/             # 16 layer experiments
│   ├── scaling_study/           # 4 scaling experiments
│   └── composite/               # 6 composite CNN experiments
├── models_pt/                   # Trained PyTorch models
├── docs/                        # Documentation
│   ├── ANALYSIS_SUMMARY.md      # Complete analysis report
│   ├── TECHNICAL_NOTES.md       # EZKL implementation details
│   └── ISSUES_LOG.md            # Hardware limitations documented
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- EZKL v7.0+ (see [EZKL installation guide](https://docs.ezkl.xyz/))

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/cnn_ezkl_bench.git
cd cnn_ezkl_bench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install EZKL (if not already installed)
# Follow instructions at https://docs.ezkl.xyz/
```

## Quick Start

### Run All Experiments

```bash
# Run all 40 experiments (core layers + scaling + composite)
python src/run_experiments.py

# This will:
# 1. Train 4 composite CNN models on CIFAR-10 (~2-3 hours)
# 2. Benchmark 24 core layer experiments (~4-6 hours)
# 3. Benchmark 8 scaling experiments (~2-3 hours)
# 4. Benchmark 8 composite experiments (~3-4 hours)
# Total estimated time: 10-15 hours on RTX 3090
```

### Run Specific Experiment Phases

```bash
# Run only core layer experiments (12 layers × 2 tolerances = 24)
python src/run_experiments.py --core-only

# Run only scaling study (4 configs × 2 tolerances = 8)
python src/run_experiments.py --scaling-only

# Run only composite architectures (4 models × 2 tolerances = 8)
# Skip training if models already exist
python src/run_experiments.py --composite-only --skip-training
```

### Analyze Results

```bash
# Generate LaTeX tables, Pareto plots, and summary report
python src/analyze_results.py
```

## Experiment Configuration

### Layer Types (8 feasible layers + 4 excluded)

**✅ Activations (4):**
- ReLU
- SiLU
- Tanh
- Polynomial (x²)

**✅ Pooling (2):**
- MaxPool2d (k=2)
- AvgPool2d (k=2)

**❌ Convolution (3) - EXCLUDED:**
- Conv2d (k=3, s=1) - OOM during proving (requires >150GB RAM)
- Conv2d (k=3, s=2) - OOM during proving (requires >150GB RAM)
- DepthwiseConv2d - OOM during proving (requires >150GB RAM)

**✅ Normalization (1):**
- BatchNorm2d

**❌ Normalization (1) - EXCLUDED:**
- LayerNorm - Excessive circuit size (logrows=24, requires >150GB RAM)

**✅ Linear (1):**
- Dense/Linear

### Precision Settings (EZKL Scale Parameters)

EZKL uses `input_scale` and `param_scale` to control fixed-point precision:
- **Accuracy Mode:** scale = 10 (higher precision, larger circuits, slower proving)
- **Efficiency Mode:** scale = 7 (lower precision, smaller circuits, faster proving)

Note: In our experimental configuration files, we use a "tolerance" parameter (0.5, 2.0) that maps to these EZKL scales. This is purely for our experimental design - EZKL v23.0.3 uses `input_scale`/`param_scale`, not "tolerance".

### Composite Architectures (3 working + 1 excluded)

**✅ Working Architectures:**

1. **CNN-Poly:** ZK-friendly with polynomial activations + AvgPool (288.8s proof)
2. **CNN-Mixed:** Hybrid approach (Poly then ReLU) (291.1s proof)
3. **CNN-Strided:** Pooling-free with strided convolutions (69.8s proof, fastest!)

**❌ Excluded Architecture:**

4. **CNN-ReLU:** Traditional CNN with ReLU + MaxPool - **OOM during proving key generation** (failed 4 attempts on 125GB RAM)
   - Reason: ReLU-based composite generates proving keys >100GB
   - Alternative: Use CNN-Mixed (includes ReLU but with Poly first layer)

## Metrics Collected

For each experiment, we measure:

| Metric | Unit | Description |
|--------|------|-------------|
| Proof time | seconds | Time to generate proof |
| Proof size | kilobytes | Size of proof artifact |
| Circuit size | constraints | Number of arithmetic constraints |
| Peak memory | gigabytes | RAM usage during proving |
| Test accuracy | % | Model accuracy on CIFAR-10 (composite only) |

## Results

Results are saved in JSON format:

```
results/
├── core_layers/
│   ├── ReLU_tol0.5_results.json
│   ├── ReLU_tol2.0_results.json
│   └── ...
├── scaling_study/
│   ├── Conv2d_Scaling_Small_tol0.5_results.json
│   └── ...
├── composite/
│   ├── CNN_ReLU_tol0.5_results.json
│   └── ...
└── experiment_log_YYYYMMDD_HHMMSS.json  # Complete log
```

### Generated Artifacts

After running analysis:

```
paper/
├── table_core_layers.tex
├── table_tolerance_comparison.tex
├── table_composite_comparison.tex
└── pareto_frontiers.png

results/
└── summary_report.txt
```

## Example Usage

### Train a Single Composite Model

```python
from src.models import create_composite_model
from src.train_cifar10 import train_composite_model
from config.experiment_config import COMPOSITE_ARCHITECTURES, TRAINING_CONFIG

# Create model
model = create_composite_model('CNN_ReLU', COMPOSITE_ARCHITECTURES['CNN_ReLU'])

# Train on CIFAR-10
results = train_composite_model(model, 'CNN_ReLU', TRAINING_CONFIG)
print(f"Test accuracy: {results['final_test_acc']:.2f}%")
```

### Benchmark a Single Layer

```python
from src.models import create_single_layer_model
from src.ezkl_utils import benchmark_model
from config.experiment_config import LAYER_CONFIGS

# Create ReLU layer
config = LAYER_CONFIGS['ReLU']
model = create_single_layer_model('ReLU', config)

# Benchmark with tolerance=0.5
metrics = benchmark_model(
    model=model,
    model_name='ReLU',
    input_shape=config['input_shape'],
    tolerance=0.5
)

print(f"Proof time: {metrics['proof_time_mean']:.2f}s")
print(f"Proof size: {metrics['proof_size_kb']:.2f} KB")
```

## Hardware Requirements

**For Core Layers Only (16 experiments):**
- GPU: Any CUDA-capable (8GB+ VRAM)
- CPU: 8+ cores
- RAM: **64GB minimum**
- Storage: 50GB

**For Complete Benchmark (26 experiments - used in paper):**
- GPU: NVIDIA RTX 3090 (24GB) or equivalent
- CPU: Intel i9-10940X (14+ cores)
- RAM: **125GB minimum** (critical bottleneck!)
- Storage: 100GB SSD

**For Conv2d/LayerNorm (10 excluded experiments - not tested):**
- RAM: **200GB+ estimated**
- May require cloud infrastructure (AWS EC2 r6i.8xlarge or similar)

**Important Notes:**
- **Bottleneck is system RAM, not GPU VRAM** (GPU often <5% utilized)
- Composite models require ~60GB RAM each during proving
- Parallel execution limited by shared RAM pool (sequential recommended)
- CPU cores more important than GPU for EZKL proving workload

## Troubleshooting

### EZKL Installation Issues

```bash
# Ensure Rust is installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install EZKL from source
cargo install ezkl
```

### CUDA Out of Memory

Reduce batch size in `config/experiment_config.py`:

```python
TRAINING_CONFIG = {
    'batch_size': 64,  # Reduced from 128
    ...
}
```

### ONNX Export Errors

Ensure model is in eval mode and uses fixed input shapes:

```python
model.eval()
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=14)
```

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{akor2025ezkl,
  title={Benchmarking CNN Components for Verifiable Inference on EVM-Compatible Blockchains: A Layer-Level Analysis of EZKL Performance},
  author={Akor, George Chidera and Ahakonye, Love Allen Chijioke and Lee, Jae Min and Kim, Dong-Seong},
  booktitle={IEEE Conference Proceedings},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This work was supported by:
- Innovative Human Resource Development for Local Intellectualization (IITP-2025-RS-2020-II201612, 33%)
- Priority Research Centers Program (2018R1A6A1A03024003, 33%)
- ITRC support program (IITP-2025-RS-2024-00438430, 34%)

## Contact

For questions or issues:
- Email: georgeakor@kumoh.ac.kr
- GitHub Issues: https://github.com/your-org/cnn_ezkl_bench/issues
