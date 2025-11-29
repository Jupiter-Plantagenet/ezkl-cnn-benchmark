# Quick Start Guide

## 1. Verify Setup

```bash
python test_setup.py
```

This will verify:
- PyTorch and CUDA
- EZKL installation
- Model creation
- ONNX export
- Experiment configuration

## 2. Run Experiments

### Option A: Run Everything (Recommended for first run)

```bash
./run_benchmark.sh all
```

This will execute all 40 experiments:
- Train 4 composite models on CIFAR-10
- Benchmark 24 core layer experiments
- Benchmark 8 scaling experiments  
- Benchmark 8 composite experiments

**Estimated time:** 10-15 hours on RTX 3090

### Option B: Run in Stages

```bash
# Stage 1: Core layers (24 experiments, ~4-6 hours)
./run_benchmark.sh core

# Stage 2: Scaling study (8 experiments, ~2-3 hours)
./run_benchmark.sh scaling

# Stage 3: Composite (8 experiments, ~3-4 hours)
./run_benchmark.sh composite
```

### Option C: Train Models Separately

```bash
# First, train the models (~2-3 hours)
./run_benchmark.sh train

# Then run benchmarks without retraining
python src/run_experiments.py --skip-training
```

## 3. Analyze Results

```bash
./run_benchmark.sh analyze
```

This generates:
- LaTeX tables for the paper (`paper/*.tex`)
- Pareto frontier plots (`paper/pareto_frontiers.png`)
- Summary report (`results/summary_report.txt`)

## 4. View Results

### Check experiment log

```bash
# Find the latest experiment log
ls -lt results/experiment_log_*.json | head -1

# View with pretty printing
python -m json.tool results/experiment_log_YYYYMMDD_HHMMSS.json
```

### Check individual results

```bash
# Core layer results
ls results/core_layers/

# Composite results
ls results/composite/

# View a specific result
cat results/core_layers/ReLU_tol0.5_results.json
```

### View summary report

```bash
cat results/summary_report.txt
```

## Troubleshooting

### "EZKL not found"

```bash
# Install EZKL
cargo install ezkl

# Or follow: https://docs.ezkl.xyz/
```

### "CUDA out of memory"

Edit `config/experiment_config.py`:

```python
TRAINING_CONFIG = {
    'batch_size': 64,  # Reduce from 128
    ...
}
```

### "Import error: No module named 'src'"

Make sure to run from the project root:

```bash
cd /home/s15/CascadeProjects/cnn_ezkl_bench
python src/run_experiments.py
```

### Clean and restart

```bash
./run_benchmark.sh clean
python test_setup.py
./run_benchmark.sh all
```

## Monitoring Progress

The experiment runner prints detailed progress:

```
======================================================================
EZKL CNN Benchmark - Automated Experiment Runner
======================================================================
Device: cuda
Results directory: ./results
Total experiments planned: 40
  - Core layers: 24
  - Scaling study: 8
  - Composite: 8
======================================================================

######################################################################
# PHASE 1: CORE LAYER EXPERIMENTS (24 experiments)
######################################################################

[1/24] Running: ReLU (tolerance=0.5)
============================================================
Benchmarking: ReLU (tolerance=0.5)
============================================================
  Exporting to ONNX...
  Generating settings (tolerance=0.5)...
  Calibrating settings...
  Compiling circuit...
  Setting up SRS...
  Generating proving and verifying keys...
  Generating witness...
  Generating proof (3 runs)...
    Run 1: 2.34s, Memory delta: 1.52 GB
    Run 2: 2.31s, Memory delta: 1.50 GB
    Run 3: 2.32s, Memory delta: 1.51 GB
  Proof time: 2.32s ± 0.01s
  Proof size: 1.45 KB
  Verifying proof...
  Verification: PASSED (0.0234s)

✓ Benchmark completed successfully
✓ Completed: ReLU (tolerance=0.5)

[2/24] Running: ReLU (tolerance=2.0)
...
```

## Expected Outputs

After completion, you should have:

```
results/
├── core_layers/           # 24 JSON files
├── scaling_study/         # 8 JSON files
├── composite/             # 8 JSON files
├── experiment_log_*.json  # Complete log
└── summary_report.txt

paper/
├── table_core_layers.tex
├── table_tolerance_comparison.tex
├── table_composite_comparison.tex
└── pareto_frontiers.png

models_pt/
└── composite/
    ├── CNN_ReLU_best.pt
    ├── CNN_Poly_best.pt
    ├── CNN_Mixed_best.pt
    └── CNN_Strided_best.pt
```

## Next Steps

1. **Review results:** Check `results/summary_report.txt`
2. **Update paper:** Copy LaTeX tables from `paper/` to your manuscript
3. **Customize:** Modify `config/experiment_config.py` for additional experiments
4. **Share:** Push results to your repository

## Getting Help

- Check `README.md` for detailed documentation
- Review error messages in the terminal output
- Inspect individual result JSON files
- Contact: georgeakor@kumoh.ac.kr
