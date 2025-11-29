#!/bin/bash
# Monitor status of all 3 parallel GPU experiments

clear
echo "========================================================================"
echo "PARALLEL EXPERIMENT STATUS - $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo ""

# Function to check process status
check_process() {
    local pid=$1
    local gpu=$2
    local phase=$3
    
    if ps -p $pid > /dev/null 2>&1; then
        runtime=$(ps -p $pid -o etime= | xargs)
        cpu=$(ps -p $pid -o %cpu= | xargs)
        mem=$(ps -p $pid -o %mem= | xargs)
        echo "✅ GPU $gpu (Phase $phase): RUNNING"
        echo "   PID: $pid | Runtime: $runtime | CPU: ${cpu}% | Mem: ${mem}%"
    else
        echo "⏹️  GPU $gpu (Phase $phase): STOPPED (PID: $pid)"
    fi
}

# Check GPU 0
if [ -f .gpu0_pid ]; then
    GPU0_PID=$(cat .gpu0_pid)
    check_process $GPU0_PID 0 1
else
    echo "❌ GPU 0: No PID file found"
fi

echo ""

# Check GPU 1
if [ -f .gpu1_pid ]; then
    GPU1_PID=$(cat .gpu1_pid)
    check_process $GPU1_PID 1 2
else
    echo "❌ GPU 1: No PID file found"
fi

echo ""

# Check GPU 2
if [ -f .gpu2_pid ]; then
    GPU2_PID=$(cat .gpu2_pid)
    check_process $GPU2_PID 2 3
else
    echo "❌ GPU 2: No PID file found"
fi

echo ""
echo "========================================================================"
echo "RESULTS COMPLETED"
echo "========================================================================"
echo ""

# Count completed experiments
PHASE1_COUNT=$(ls -1 results/core_layers/LayerNorm*.json results/core_layers/Dense*.json 2>/dev/null | wc -l)
PHASE2_COUNT=$(ls -1 results/scaling_study/*.json 2>/dev/null | wc -l)
PHASE3_COUNT=$(ls -1 results/composite/*.json 2>/dev/null | wc -l)

echo "Phase 1 (GPU 0): $PHASE1_COUNT / 4 experiments"
if [ $PHASE1_COUNT -gt 0 ]; then
    echo "  Latest:"
    ls -t results/core_layers/LayerNorm*.json results/core_layers/Dense*.json 2>/dev/null | head -2 | xargs -I {} sh -c 'echo "    - $(basename {}) ($(stat -c %y {} | cut -d. -f1))"'
fi

echo ""
echo "Phase 2 (GPU 1): $PHASE2_COUNT / 4 experiments"
if [ $PHASE2_COUNT -gt 0 ]; then
    echo "  Latest:"
    ls -t results/scaling_study/*.json 2>/dev/null | head -2 | xargs -I {} sh -c 'echo "    - $(basename {}) ($(stat -c %y {} | cut -d. -f1))"'
fi

echo ""
echo "Phase 3 (GPU 2): $PHASE3_COUNT / 8 experiments"
if [ $PHASE3_COUNT -gt 0 ]; then
    echo "  Latest:"
    ls -t results/composite/*.json 2>/dev/null | head -2 | xargs -I {} sh -c 'echo "    - $(basename {}) ($(stat -c %y {} | cut -d. -f1))"'
fi

echo ""
echo "========================================================================"
echo "GPU UTILIZATION"
echo "========================================================================"
echo ""
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{printf "GPU %s: %s%% compute, %s%% memory (%s / %s MB)\n", $1, $3, $4, $5, $6}'

echo ""
echo "========================================================================"
echo "To view logs:"
echo "  tail -f experiment_run_gpu0.log"
echo "  tail -f experiment_run_gpu1.log"
echo "  tail -f experiment_run_gpu2.log"
echo "========================================================================"
