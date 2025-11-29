# Issues Log - CNN EZKL Benchmark

## Date: November 29, 2025

### Issue 1: EZKL API Compatibility (v23.0.3)

**Problem:** Initial implementation used non-existent `tolerance` attribute on `PyRunArgs`.

**Error:**
```
'builtins.PyRunArgs' object has no attribute 'tolerance'
```

**Root Cause:** 
- EZKL v23.0.3 doesn't have a `tolerance` parameter
- Documentation mismatch - we assumed tolerance existed based on paper design

**Solution:**
- Use `input_scale` and `param_scale` instead
- Map our experimental "tolerance" parameter to EZKL scales:
  - tolerance 0.5 → scale 10 (accuracy mode)
  - tolerance 2.0 → scale 7 (efficiency mode)

**Files Modified:**
- `src/ezkl_utils.py` - Changed to use `run_args.input_scale` and `run_args.param_scale`
- Added proper documentation explaining the mapping

**Impact:** All 40 initial experiments failed before fix was applied.

---

### Issue 2: Calibration JSON Deserialization Failure

**Problem:** EZKL calibration failed with JSON parsing error.

**Error:**
```
Failed to calibrate settings: [graph] [json] failed to deserialize FileSourceInner at line 1 column 84657
```

**Root Cause:**
- Input data format incompatibility
- EZKL expected flattened 1D arrays, we provided nested arrays

**Solution:**
- Flatten input data before saving to JSON: `dummy_input.detach().numpy().reshape(-1).tolist()`
- Made calibration optional (non-critical step - can continue without it)

**Files Modified:**
- `src/ezkl_utils.py` - Lines 63, 114-120

**Impact:** Calibration now works for simple layers, fails gracefully for complex ones.

---

### Issue 3: Async Event Loop Error in SRS Setup

**Problem:** `get_srs()` failed with event loop error.

**Error:**
```
no running event loop
```

**Root Cause:**
- `ezkl.get_srs()` attempts to download SRS from remote server asynchronously
- No event loop available in synchronous context

**Solution:**
- Changed from `get_srs()` to `gen_srs()` 
- Generate SRS locally instead of downloading
- Slower but works without async complications

**Files Modified:**
- `src/ezkl_utils.py` - Line 154

**Impact:** Added ~10-30 seconds per experiment for SRS generation.

---

### Issue 4: Conv2d Out-of-Memory Termination

**Problem:** Process killed while generating Conv2d circuits.

**Symptoms:**
- Process terminated at Conv2d_k3_s1 experiment
- Last log entry: "Generating SRS with logrows=22..."
- System had 125GB RAM but still ran out of memory

**Details:**
- Input shape: (1, 32, 64, 64) - Batch=1, Channels=32, H=64, W=64
- Circuit complexity: logrows=22 → 4,194,304 circuit rows
- Estimated proving key size: 50-100+ GB (AvgPool2d was 8.6 GB)
- Swap space (2GB) was exhausted

**Successful Experiments Before Crash:**
- ✅ ReLU (2 runs)
- ✅ SiLU (2 runs)
- ✅ Tanh (2 runs)
- ✅ Poly (2 runs)
- ✅ MaxPool2d (2 runs)
- ✅ AvgPool2d (2 runs)
Total: 12/40 experiments completed successfully

**Failed Experiment:**
- ❌ Conv2d_k3_s1 (tolerance=0.5) - OOM during SRS generation

**Analysis:**
- Simple layers (activations, pooling): 1-10 GB proving keys
- Conv2d layers: 50-100+ GB proving keys (estimated)
- Memory requirement scales exponentially with:
  - Input spatial dimensions (H × W)
  - Number of channels
  - Kernel size
  - EZKL scale (precision)

**Affected Experiments (10 total):**
- Conv2d_k3_s1 (2) - Standard 3×3 convolution
- Conv2d_k3_s2 (2) - Strided convolution
- DepthwiseConv2d (2) - Depthwise separable
- Conv2d_Scaling_Small (2) - Scaling study
- Conv2d_Scaling_Large (2) - Scaling study

**Potential Solutions:**
1. **Reduce input size:** (1, 16, 32, 32) instead of (1, 32, 64, 64) - 4x reduction
2. **Use only scale=7:** Skip accuracy mode for Conv2d
3. **Cloud instance:** Run Conv2d on 256GB+ RAM machine
4. **Skip Conv2d:** Focus on completable layers (current approach)

**Decision:** Skip Conv2d experiments temporarily, complete other layers first.

**Impact:** 
- 10 Conv2d experiments postponed
- 30 remaining experiments should complete successfully
- Paper will need to note Conv2d memory constraints

---

### Issue 5: Output Buffering in Background Process

**Problem:** Log file appeared "stuck" for 30+ minutes with no updates.

**Symptoms:**
- `experiment_run.log` not updating
- Process still running (high CPU usage)
- No visible progress

**Root Cause:**
- Python stdout buffering when redirected to file with `nohup ... > log.txt`
- Conv2d experiments take 30-60 minutes each
- Buffered output not flushed until completion or crash

**Solution:**
- Process was actually working correctly
- Could monitor via temp files: `ls -lht temp/core_layers/`
- Consider adding `python -u` for unbuffered output in future runs

**Impact:** User uncertainty about progress; no actual failure.

---

## Summary of Working Solutions

### EZKL API Usage (v23.0.3)
```python
# ✅ CORRECT
run_args = ezkl.PyRunArgs()
run_args.input_scale = 10  # or 7
run_args.param_scale = 10  # or 7
run_args.input_visibility = "public"
run_args.output_visibility = "public"
run_args.param_visibility = "private"

# Use gen_srs, not get_srs
ezkl.gen_srs(srs_path=str(srs_path), logrows=logrows)

# Flatten input data
input_data = dummy_input.detach().numpy().reshape(-1).tolist()
```

### Memory Constraints
- Simple layers: < 10 GB proving keys ✅
- Conv2d layers: 50-100+ GB proving keys ❌ (OOM on 125GB system)
- Need input reduction or larger instance for Conv2d

---

## Remaining Work

### Completed (12/40)
- ✅ Activations: ReLU, SiLU, Tanh, Poly
- ✅ Pooling: MaxPool2d, AvgPool2d

### In Progress (0/40)
- (None - process was killed)

### Postponed (10/40)
- ⏸️ Conv2d_k3_s1
- ⏸️ Conv2d_k3_s2
- ⏸️ DepthwiseConv2d
- ⏸️ Conv2d scaling studies (4 experiments)

### To Run Next (18/40)
- 🔲 BatchNorm2d (2)
- 🔲 LayerNorm (2)
- 🔲 Dense/Linear (2)
- 🔲 Dense scaling studies (4)
- 🔲 Composite models (8)

---

## Next Steps

1. ✅ Document all issues (this file)
2. 🔲 Create modified experiment runner skipping Conv2d
3. 🔲 Run remaining 18 experiments
4. 🔲 Analyze partial results
5. 🔲 Return to Conv2d with reduced input sizes or cloud instance

---

## Hardware Specs

- CPU: Intel i9-10940X
- **System RAM: 125 GB (SHARED across all processes and GPUs)**
- Swap: 2 GB
- **GPU 0:** NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **GPU 1:** NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **GPU 2:** NVIDIA GeForce RTX 3090 (24 GB VRAM)
- Storage: 4.6 TB free
- OS: Linux

**Critical Architecture Note:** 
- The 125GB RAM is **SYSTEM RAM** shared by ALL processes, ALL 3 GPUs
- Each GPU has its own 24GB VRAM (GPU memory), separate from system RAM
- EZKL proof generation bottleneck is **system RAM**, not GPU VRAM
- Multiple processes compete for the same 125GB RAM pool

**Note:** Even with 125GB RAM, Conv2d circuits exceed available memory. Conv2d proving key estimated at 50-100GB+ is too large for this system.

---

## Issue #6: LayerNorm Excessive Circuit Size

**Date:** November 29, 2025, 19:20 KST

### Problem
LayerNorm experiment generated extremely large circuit requiring logrows=24 (16,777,216 rows), causing SRS generation to take 15+ minutes and key generation to likely exceed memory limits.

### Symptoms
- Circuit compilation succeeded
- SRS generation with logrows=24 took 10+ minutes
- Process appeared stuck on proving key generation
- High CPU usage (1800%+) but very slow progress
- Estimated proving key size: 50-100+ GB

### Root Cause
LayerNorm's normalization operations require high precision and many constraints, resulting in massive circuit size similar to Conv2d layers.

### Impact
- **Experiments affected:** LayerNorm (2 experiments: tol=0.5, tol=2.0)
- **Blocking:** Phase 1 core layer experiments on GPU 0

### Solution
Postponed LayerNorm experiments similar to Conv2d. Will require either:
1. Reduced input dimensions
2. Hardware with more RAM (256GB+)
3. Different EZKL configuration/optimization

### Files Modified
- Stopped GPU 0 process (PID: 2508292) after 10+ minutes on single experiment

---

## Issue #7: Composite Model OOM During Key Generation (GPU 1)

**Date:** November 29, 2025, 20:05 KST

### Problem
CNN_ReLU composite model (logrows=23, ~8.4M circuit rows) process was killed during proving key generation phase on GPU 1.

### Symptoms
- Calibration completed successfully
- Circuit compiled successfully
- SRS generation with logrows=23 completed
- Process killed during "Generating proving and verifying keys..." phase
- No error message, process silently stopped

### Root Cause
Proving key generation for composite CNN models requires significant memory. Key generation phase appeared to exceed available RAM (125GB system + 24GB GPU), triggering OOM kill.

### Impact
- **Experiments affected:** CNN_ReLU (tol=0.5) on GPU 1
- **Status:** Process restarted automatically, will retry

### Workaround
Process was restarted and will retry. If issue persists, may need to:
1. Run composite models sequentially instead of parallel
2. Use machine with more RAM
3. Investigate EZKL memory optimization settings

### Files Modified
- GPU 1 process (PID: 2521603) killed and restarted (new PID: 2550225)
- Logs: `experiment_run_composite_gpu1.log`

---

## Issue #8: Dense Experiments Postponed

**Date:** November 29, 2025, 19:31 KST

### Problem
Dense (linear) layer experiments were not included in the composite GPU runner scripts, leaving 4 core layer experiments incomplete.

### Impact
- **Experiments affected:** Dense (2 experiments: tol=0.5, tol=2.0)
- **Status:** Will need separate run after composite experiments complete

### Solution
Dense layer experiments deferred until after composite model experiments. These should complete quickly (smaller circuits than CNN models).

---

## Summary of Postponed/Incomplete Experiments

### Completed: 19/30 non-Conv2d experiments (63%)
- ✅ Phase 1 Core Layers: 14/18
  - ReLU, SiLU, Tanh, Poly (8)
  - MaxPool2d, AvgPool2d (4)
  - BatchNorm2d (2)
- ✅ Phase 2 Scaling Study: 4/4
  - Dense_Scaling_Small (2)
  - Dense_Scaling_Large (2)
- 🔄 Phase 3 Composite: 1/8
  - CNN_Mixed (tol=0.5) ✅

### In Progress: 7 experiments
- CNN_Mixed (tol=2.0) - GPU 2
- CNN_ReLU (2) - GPU 1 retrying
- CNN_Poly (2)
- CNN_Strided (2)

### Postponed - Memory Constraints: 14 experiments
- ⏸️ Conv2d_k3_s1 (2)
- ⏸️ Conv2d_k3_s2 (2)
- ⏸️ DepthwiseConv2d (2)
- ⏸️ Conv2d_Scaling_Small (2)
- ⏸️ Conv2d_Scaling_Large (2)
- ⏸️ LayerNorm (2) - **NEW**
- ⏸️ Dense (2) - Will run later

### Total Experiment Count
- **Original plan:** 40 experiments (12 core layers × 2 scales + 4 scaling configs × 2 + 4 composite × 2)
- **Feasible on current hardware:** 26 experiments (excluding Conv2d variants and LayerNorm)
- **Target for paper:** 26 experiments
- **Completed so far:** 20
- **Remaining:** 6 (composite models + Dense)

---

## Issue #9: Composite Models Require Sequential Execution

**Date:** November 29, 2025, 20:30 KST

### Problem
Attempted to run composite CNN experiments in parallel on GPU 1 and GPU 2, but GPU 1 process was repeatedly OOM killed during proving key generation.

### Symptoms
- GPU 2 process successfully running CNN_Mixed experiments
- GPU 1 process killed twice during CNN_ReLU proving key generation
- System RAM usage peaked at ~74GB with GPU 2 alone
- Free RAM dropped to 13GB (critically low)
- Swap fully utilized (2GB)

### Root Cause
**System RAM Architecture Misunderstanding:**
- The 125GB RAM is **SHARED** across all 3 GPUs and all processes
- Each composite model proof generation requires ~60GB system RAM
- Running 2 composite experiments in parallel: 2 × 60GB = 120GB > available
- Proof generation is CPU-intensive and RAM-intensive, not GPU-intensive
- GPU VRAM (24GB per GPU) is separate and not the bottleneck

### Measurements
- **CNN_Mixed (tol=0.5):** Peak RAM usage ~60GB
- **CNN_Mixed (tol=2.0):** Peak RAM usage ~62GB during proof generation
- **System overhead:** ~12-15GB
- **Available for experiments:** ~110GB
- **Required for 2 parallel composite:** ~120GB ❌

### Impact
- **Experiments affected:** All composite model experiments must run sequentially
- **Time impact:** 3-4 hours total instead of 1.5-2 hours if parallel worked
- **GPU utilization:** GPUs idle during proof generation (CPU/RAM bottleneck)

### Solution
Run composite experiments **sequentially**, one at a time:
1. ✅ CNN_Mixed (tol=0.5, 2.0) - Completed on GPU 2
2. 🔄 CNN_Strided (tol=0.5, 2.0) - Next on GPU 2
3. Next: CNN_ReLU (tol=0.5, 2.0) - Then on GPU 1 or 2
4. Next: CNN_Poly (tol=0.5, 2.0) - Then on GPU 1 or 2

**Meanwhile:** Dense layer experiments can run on idle GPU since they use much less RAM (~5-10GB estimated).

### Key Learning
When running EZKL experiments:
- **Bottleneck is system RAM**, not GPU VRAM
- Large circuits (logrows=23+) need 50-100GB system RAM per experiment
- Must account for shared RAM pool when planning parallel execution
- GPU assignment matters less than RAM availability
