#!/bin/bash
# Monitor status of composite experiment GPUs

clear
echo "========================================================================"
echo "COMPOSITE EXPERIMENT STATUS - $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo ""

# Check GPU 1
if [ -f .composite_gpu1_pid ]; then
    GPU1_PID=$(cat .composite_gpu1_pid)
    if ps -p $GPU1_PID > /dev/null 2>&1; then
        runtime=$(ps -p $GPU1_PID -o etime= | xargs)
        cpu=$(ps -p $GPU1_PID -o %cpu= | xargs)
        mem=$(ps -p $GPU1_PID -o %mem= | xargs)
        echo "✅ GPU 1 (CNN_ReLU, CNN_Poly): RUNNING"
        echo "   PID: $GPU1_PID | Runtime: $runtime | CPU: ${cpu}% | Mem: ${mem}%"
    else
        echo "⏹️  GPU 1: STOPPED (PID: $GPU1_PID)"
    fi
else
    echo "❌ GPU 1: No PID file found"
fi

echo ""

# Check GPU 2
if [ -f .composite_gpu2_pid ]; then
    GPU2_PID=$(cat .composite_gpu2_pid)
    if ps -p $GPU2_PID > /dev/null 2>&1; then
        runtime=$(ps -p $GPU2_PID -o etime= | xargs)
        cpu=$(ps -p $GPU2_PID -o %cpu= | xargs)
        mem=$(ps -p $GPU2_PID -o %mem= | xargs)
        echo "✅ GPU 2 (CNN_Mixed, CNN_Strided): RUNNING"
        echo "   PID: $GPU2_PID | Runtime: $runtime | CPU: ${cpu}% | Mem: ${mem}%"
    else
        echo "⏹️  GPU 2: STOPPED (PID: $GPU2_PID)"
    fi
else
    echo "❌ GPU 2: No PID file found"
fi

echo ""
echo "========================================================================"
echo "RESULTS COMPLETED"
echo "========================================================================"
echo ""

COMPOSITE_COUNT=$(ls -1 results/composite/*.json 2>/dev/null | wc -l)
echo "Composite Models: $COMPOSITE_COUNT / 8 experiments"

if [ $COMPOSITE_COUNT -gt 0 ]; then
    echo ""
    echo "Latest completions:"
    ls -t results/composite/*.json 2>/dev/null | head -4 | while read file; do
        name=$(basename "$file" | sed 's/_results.json//')
        time=$(stat -c %y "$file" | cut -d. -f1)
        echo "  ✓ $name ($time)"
    done
fi

echo ""
echo "========================================================================"
echo "GPU UTILIZATION"
echo "========================================================================"
echo ""
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | awk -F, '{printf "GPU %s: %s%% compute, %s%% memory (%s / %s MB)\n", $1, $3, $4, $5, $6}'

echo ""
echo "========================================================================"
echo "LOGS (last 10 lines each):"
echo "========================================================================"
echo ""
echo "--- GPU 1 ---"
tail -10 experiment_run_composite_gpu1.log 2>/dev/null | grep -v "^$"
echo ""
echo "--- GPU 2 ---"
tail -10 experiment_run_composite_gpu2.log 2>/dev/null | grep -v "^$"
echo ""
echo "========================================================================"
echo "Monitor: watch -n 30 './check_composite_status.sh'"
echo "========================================================================"
