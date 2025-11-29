# Parallel GPU Experiment Run Summary

## Setup Date: November 29, 2025, 19:15 KST

## Overview

This document describes the parallel experiment setup using 3 RTX 3090 GPUs to complete the remaining 16 experiments.

## Hardware Configuration

- **GPU 0:** NVIDIA GeForce RTX 3090 (24GB) - Phase 1
- **GPU 1:** NVIDIA GeForce RTX 3090 (24GB) - Phase 2
- **GPU 2:** NVIDIA GeForce RTX 3090 (24GB) - Phase 3
- **System RAM:** 125 GB
- **CPU:** Intel i9-10940X

## Experiment Distribution

### GPU 0: Phase 1 - Remaining Core Layers (4 experiments)
**Script:** `run_phase1_gpu0.py`  
**Work Dir:** `temp_gpu0/`  
**Log:** `experiment_run_gpu0.log`  
**Estimated Time:** 45-60 minutes

**Experiments:**
1. LayerNorm (tol=0.5)
2. LayerNorm (tol=2.0)
3. Dense (tol=0.5)
4. Dense (tol=2.0)

### GPU 1: Phase 2 - Dense Scaling Studies (4 experiments)
**Script:** `run_phase2_gpu1.py`  
**Work Dir:** `temp_gpu1/`  
**Log:** `experiment_run_gpu1.log`  
**Estimated Time:** 1.5-2 hours

**Experiments:**
1. Dense_Scaling_Small (tol=0.5)
2. Dense_Scaling_Small (tol=2.0)
3. Dense_Scaling_Large (tol=0.5)
4. Dense_Scaling_Large (tol=2.0)

### GPU 2: Phase 3 - Composite Models (8 experiments)
**Script:** `run_phase3_gpu2.py`  
**Work Dir:** `temp_gpu2/`  
**Log:** `experiment_run_gpu2.log`  
**Estimated Time:** 2-3 hours

**Experiments:**
1. CNN_ReLU (tol=0.5)
2. CNN_ReLU (tol=2.0)
3. CNN_Poly (tol=0.5)
4. CNN_Poly (tol=2.0)
5. CNN_Mixed (tol=0.5)
6. CNN_Mixed (tol=2.0)
7. CNN_Strided (tol=0.5)
8. CNN_Strided (tol=2.0)

## Already Completed (14 experiments)

From previous run (results preserved):
- ReLU (2)
- SiLU (2)
- Tanh (2)
- Poly (2)
- MaxPool2d (2)
- AvgPool2d (2)
- BatchNorm2d (2)

## Postponed Experiments (10 experiments)

Due to memory constraints (Conv2d requires 50-100+ GB proving keys):
- Conv2d_k3_s1 (2)
- Conv2d_k3_s2 (2)
- DepthwiseConv2d (2)
- Conv2d_Scaling_Small (2)
- Conv2d_Scaling_Large (2)

## Safety Features

### No Duplicate Work
- Each script checks if result files already exist before running
- Skips experiments that are already completed
- Results directory is shared, temp directories are separate

### No Interference
- Each GPU has dedicated `CUDA_VISIBLE_DEVICES` set
- Separate work directories (`temp_gpu0/`, `temp_gpu1/`, `temp_gpu2/`)
- Separate log files
- Results written to shared directory but different subdirectories

### Monitoring
- `./check_parallel_status.sh` - Shows status of all 3 processes
- Individual log files for each GPU
- PID tracking in `.gpu*_pid` files
- Automatic backups after each phase completes

## Launch Commands

### Start All 3 in Parallel
```bash
./run_parallel_experiments.sh
```

### Monitor Progress
```bash
watch -n 30 ./check_parallel_status.sh
```

### View Individual Logs
```bash
tail -f experiment_run_gpu0.log  # GPU 0 - Phase 1
tail -f experiment_run_gpu1.log  # GPU 1 - Phase 2
tail -f experiment_run_gpu2.log  # GPU 2 - Phase 3
```

### Manual Launch (if needed)
```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 nohup python -u run_phase1_gpu0.py > experiment_run_gpu0.log 2>&1 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 nohup python -u run_phase2_gpu1.py > experiment_run_gpu1.log 2>&1 &

# GPU 2
CUDA_VISIBLE_DEVICES=2 nohup python -u run_phase3_gpu2.py > experiment_run_gpu2.log 2>&1 &
```

## Expected Results

### Timeline
- **Start:** 19:15 KST
- **GPU 0 finish:** ~20:00 KST (45 min)
- **GPU 1 finish:** ~21:00 KST (1.5 hr)
- **GPU 2 finish:** ~22:00 KST (2.5 hr)
- **All complete:** ~22:00 KST

### Final Experiment Count
- **Previously completed:** 14 experiments
- **New from parallel run:** 16 experiments
- **Total successful:** 30 experiments
- **Postponed (Conv2d):** 10 experiments
- **Grand total planned:** 40 experiments

## Verification After Completion

```bash
# Check all results
echo "Core layers: $(ls -1 results/core_layers/*.json | wc -l)"      # Expect: 18
echo "Scaling study: $(ls -1 results/scaling_study/*.json | wc -l)"  # Expect: 4
echo "Composite: $(ls -1 results/composite/*.json | wc -l)"          # Expect: 8
echo "Total: $((18 + 4 + 8)) / 30"                                   # Expect: 30
```

## Notes

1. **Unbuffered Output:** Using `python -u` flag for real-time log updates
2. **Pre-trained Models:** All composite models already trained and saved in `models_pt/`
3. **Automatic Backups:** Each runner creates a backup after completing its phase
4. **Error Handling:** Each experiment wrapped in try-catch, failures logged but don't stop progress
5. **Memory Safety:** Each experiment stays well under 24GB GPU memory and uses ~10-20GB system RAM

## Issue Log Reference

See `ISSUES_LOG.md` for detailed documentation of:
- EZKL API compatibility fixes
- Conv2d memory constraints
- All previous debugging and solutions
