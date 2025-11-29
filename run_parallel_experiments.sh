#!/bin/bash
# Master script to launch all 3 GPU runners in parallel

echo "========================================================================"
echo "PARALLEL EXPERIMENT RUNNER - 3 GPUs"
echo "========================================================================"
echo ""
echo "GPU 0: Phase 1 - LayerNorm (2), Dense (2) = 4 experiments"
echo "GPU 1: Phase 2 - Dense Scaling (4) = 4 experiments"
echo "GPU 2: Phase 3 - Composite Models (8) = 8 experiments"
echo ""
echo "Total: 16 experiments across 3 GPUs"
echo "========================================================================"
echo ""

# Check if any existing processes
if pgrep -f "run_phase.*_gpu" > /dev/null; then
    echo "⚠️  WARNING: Existing GPU experiment processes detected!"
    echo "Processes:"
    ps aux | grep "run_phase.*_gpu" | grep -v grep
    echo ""
    read -p "Kill existing processes and continue? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "run_phase.*_gpu"
        sleep 2
    else
        echo "Aborted."
        exit 1
    fi
fi

# Create temp directories
mkdir -p temp_gpu0 temp_gpu1 temp_gpu2

# Make scripts executable
chmod +x run_phase1_gpu0.py run_phase2_gpu1.py run_phase3_gpu2.py

echo "Starting GPU 0 (Phase 1)..."
nohup python -u run_phase1_gpu0.py > experiment_run_gpu0.log 2>&1 &
GPU0_PID=$!
echo "  PID: $GPU0_PID"

echo "Starting GPU 1 (Phase 2)..."
nohup python -u run_phase2_gpu1.py > experiment_run_gpu1.log 2>&1 &
GPU1_PID=$!
echo "  PID: $GPU1_PID"

echo "Starting GPU 2 (Phase 3)..."
nohup python -u run_phase3_gpu2.py > experiment_run_gpu2.log 2>&1 &
GPU2_PID=$!
echo "  PID: $GPU2_PID"

echo ""
echo "========================================================================"
echo "ALL 3 PROCESSES STARTED"
echo "========================================================================"
echo ""
echo "Process IDs:"
echo "  GPU 0: $GPU0_PID (log: experiment_run_gpu0.log)"
echo "  GPU 1: $GPU1_PID (log: experiment_run_gpu1.log)"
echo "  GPU 2: $GPU2_PID (log: experiment_run_gpu2.log)"
echo ""
echo "Monitor with:"
echo "  watch -n 30 './check_parallel_status.sh'"
echo ""
echo "Or check individual logs:"
echo "  tail -f experiment_run_gpu0.log"
echo "  tail -f experiment_run_gpu1.log"
echo "  tail -f experiment_run_gpu2.log"
echo ""
echo "========================================================================"

# Save PIDs to file for monitoring
echo "$GPU0_PID" > .gpu0_pid
echo "$GPU1_PID" > .gpu1_pid
echo "$GPU2_PID" > .gpu2_pid

echo "PIDs saved to .gpu*_pid files"
echo ""
