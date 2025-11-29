#!/usr/bin/env python
"""
GPU 0: Phase 1 Remaining Core Layers
Runs: LayerNorm (2), Dense (2) = 4 experiments
"""

import os
import sys
from pathlib import Path

# Force GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.run_experiments import ExperimentRunner
from config.experiment_config import LAYER_CONFIGS, TOLERANCE_VALUES

# Only run these layers
TARGET_LAYERS = ['LayerNorm', 'Dense']


class Phase1Runner(ExperimentRunner):
    """Runner for Phase 1 remaining layers only"""
    
    def __init__(self):
        super().__init__()
        # Override work directory to avoid conflicts
        self.work_dir = Path('temp_gpu0')
        self.work_dir.mkdir(exist_ok=True)
        
    def run_core_layer_experiments(self):
        """Run only LayerNorm and Dense experiments"""
        print(f"\n{'#'*70}")
        print(f"# GPU 0: PHASE 1 REMAINING LAYERS")
        print(f"# Target: LayerNorm (2), Dense (2)")
        print(f"# Work dir: {self.work_dir}")
        print(f"{'#'*70}\n")
        
        exp_count = 0
        completed = 0
        skipped = 0
        
        for layer_name in TARGET_LAYERS:
            if layer_name not in LAYER_CONFIGS:
                print(f"⚠️  Warning: {layer_name} not in config")
                continue
                
            config = LAYER_CONFIGS[layer_name]
            
            for tolerance in TOLERANCE_VALUES:
                exp_count += 1
                
                # Check if already completed
                result_file = self.results_dir / 'core_layers' / f'{layer_name}_tol{tolerance}_results.json'
                if result_file.exists():
                    print(f"⏭️  SKIPPED: {layer_name} (tol={tolerance}) - already completed")
                    skipped += 1
                    continue
                
                print(f"\n[GPU0-{exp_count}/4] Running: {layer_name} (tolerance={tolerance})\n")
                
                try:
                    from src.models import create_single_layer_model
                    from src.ezkl_utils import benchmark_model
                    
                    # Create model
                    model = create_single_layer_model(layer_name, config)
                    
                    # Run benchmark with GPU0-specific work dir
                    metrics = benchmark_model(
                        model=model,
                        model_name=layer_name,
                        input_shape=config['input_shape'],
                        tolerance=tolerance,
                        work_dir=self.work_dir / 'core_layers',
                        results_dir=self.results_dir / 'core_layers'
                    )
                    
                    # Log result
                    self.experiment_log.append({
                        'experiment_id': f"core_{layer_name}_tol{tolerance}",
                        'phase': 'core_layers',
                        'gpu': 0,
                        'layer_name': layer_name,
                        'tolerance': tolerance,
                        'status': metrics.get('status', 'unknown'),
                        'proof_time': metrics.get('proof_time_mean'),
                        'proof_size_kb': metrics.get('proof_size_kb'),
                        'circuit_size': metrics.get('circuit_size')
                    })
                    
                    completed += 1
                    print(f"✓ Completed: {layer_name} (tolerance={tolerance})")
                    
                except Exception as e:
                    print(f"✗ Failed: {layer_name} (tolerance={tolerance})")
                    print(f"  Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    self.experiment_log.append({
                        'experiment_id': f"core_{layer_name}_tol{tolerance}",
                        'phase': 'core_layers',
                        'gpu': 0,
                        'layer_name': layer_name,
                        'tolerance': tolerance,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        print(f"\n{'='*70}")
        print(f"GPU 0 Phase 1 Summary:")
        print(f"  Completed: {completed}")
        print(f"  Skipped: {skipped}")
        print(f"  Total processed: {exp_count}")
        print(f"{'='*70}\n")
        
        # Backup after completion
        self._backup_progress("gpu0_phase1_complete")


if __name__ == '__main__':
    print("="*70)
    print("GPU 0: PHASE 1 REMAINING CORE LAYERS")
    print("="*70)
    print("\nGPU Assignment: CUDA_VISIBLE_DEVICES=0")
    print("Work Directory: temp_gpu0/")
    print("Log File: experiment_run_gpu0.log")
    print("\nExperiments to run:")
    print("  - LayerNorm (tol=0.5)")
    print("  - LayerNorm (tol=2.0)")
    print("  - Dense (tol=0.5)")
    print("  - Dense (tol=2.0)")
    print("="*70 + "\n")
    
    runner = Phase1Runner()
    runner.run_core_layer_experiments()
    
    # Save final results
    log_data = runner.save_experiment_log()
    runner.print_summary()
    
    print("\n" + "="*70)
    print("GPU 0 PHASE 1 COMPLETE")
    print("="*70)
