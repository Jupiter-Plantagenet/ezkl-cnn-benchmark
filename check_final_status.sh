#!/bin/bash
# Monitor final composite experiments

clear
echo "========================================================================"
echo "FINAL EXPERIMENTS STATUS - $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo ""

# Check process
if [ -f .final_composite_pid ]; then
    PID=$(cat .final_composite_pid)
    if ps -p $PID > /dev/null 2>&1; then
        runtime=$(ps -p $PID -o etime= | xargs)
        cpu=$(ps -p $PID -o %cpu= | xargs)
        mem=$(ps -p $PID -o %mem= | xargs)
        echo "✅ Final Composite Experiments: RUNNING"
        echo "   PID: $PID | Runtime: $runtime | CPU: ${cpu}% | Mem: ${mem}%"
    else
        echo "⏹️  Final Composite: STOPPED (PID: $PID)"
    fi
else
    echo "❌ No PID file found"
fi

echo ""
echo "========================================================================"
echo "EXPERIMENT PROGRESS"
echo "========================================================================"
echo ""

TOTAL=$(ls -1 results/*/*.json 2>/dev/null | wc -l)
echo "✅ Completed: $TOTAL / 26 experiments"
echo ""

COMPOSITE=$(ls -1 results/composite/*.json 2>/dev/null | wc -l)
echo "Composite Models: $COMPOSITE / 8"
if [ $COMPOSITE -gt 0 ]; then
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
echo "SYSTEM RESOURCES"
echo "========================================================================"
echo ""
free -h | grep "Mem:"
echo ""
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | awk -F, '{printf "GPU %s: %s%% utilization, %s used\n", $1, $2, $3}'

echo ""
echo "========================================================================"
echo "RECENT LOG OUTPUT (last 10 lines):"
echo "========================================================================"
echo ""
tail -10 experiment_run_final_composite.log 2>/dev/null | grep -v "^$"

echo ""
echo "========================================================================"
echo "Monitor: watch -n 30 './check_final_status.sh'"
echo "Full log: tail -f experiment_run_final_composite.log"
echo "========================================================================"
