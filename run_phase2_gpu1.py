#!/usr/bin/env python
"""
GPU 1: Phase 2 Scaling Study (Dense only)
Runs: Dense_Scaling_Small (2), Dense_Scaling_Large (2) = 4 experiments
"""

import os
import sys
from pathlib import Path

# Force GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.run_experiments import ExperimentRunner
from config.experiment_config import SCALING_CONFIGS, TOLERANCE_VALUES, LAYER_CONFIGS

# Only run Dense scaling (skip Conv2d scaling)
TARGET_CONFIGS = ['Dense_Scaling_Small', 'Dense_Scaling_Large']


class Phase2Runner(ExperimentRunner):
    """Runner for Phase 2 Dense scaling studies only"""
    
    def __init__(self):
        super().__init__()
        # Override work directory to avoid conflicts
        self.work_dir = Path('temp_gpu1')
        self.work_dir.mkdir(exist_ok=True)
        
    def run_scaling_experiments(self):
        """Run Dense scaling experiments only"""
        print(f"\n{'#'*70}")
        print(f"# GPU 1: PHASE 2 SCALING STUDY (Dense only)")
        print(f"# Target: Dense_Scaling_Small (2), Dense_Scaling_Large (2)")
        print(f"# Work dir: {self.work_dir}")
        print(f"{'#'*70}\n")
        
        exp_count = 0
        completed = 0
        skipped = 0
        
        for config_name in TARGET_CONFIGS:
            if config_name not in SCALING_CONFIGS:
                print(f"⚠️  Warning: {config_name} not in config")
                continue
            
            config_spec = SCALING_CONFIGS[config_name]
            
            for tolerance in TOLERANCE_VALUES:
                exp_count += 1
                
                # Check if already completed
                result_file = self.results_dir / 'scaling_study' / f'{config_name}_tol{tolerance}_results.json'
                if result_file.exists():
                    print(f"⏭️  SKIPPED: {config_name} (tol={tolerance}) - already completed")
                    skipped += 1
                    continue
                
                print(f"\n[GPU1-{exp_count}/4] Running: {config_name} (tolerance={tolerance})\n")
                
                try:
                    from src.models import create_single_layer_model
                    from src.ezkl_utils import benchmark_model
                    
                    # Get base layer config
                    base_layer = config_spec['base_layer']
                    base_config = LAYER_CONFIGS[base_layer]
                    
                    # Create modified config
                    modified_config = {
                        'type': base_config['type'],
                        'input_shape': config_spec['input_shape'],
                        'params': config_spec['params']
                    }
                    
                    # Create model
                    model = create_single_layer_model(base_layer, modified_config)
                    
                    # Run benchmark with GPU1-specific work dir
                    metrics = benchmark_model(
                        model=model,
                        model_name=config_name,
                        input_shape=config_spec['input_shape'],
                        tolerance=tolerance,
                        work_dir=self.work_dir / 'scaling_study',
                        results_dir=self.results_dir / 'scaling_study'
                    )
                    
                    # Log result
                    self.experiment_log.append({
                        'experiment_id': f"scaling_{config_name}_tol{tolerance}",
                        'phase': 'scaling_study',
                        'gpu': 1,
                        'config_name': config_name,
                        'base_layer': base_layer,
                        'tolerance': tolerance,
                        'status': metrics.get('status', 'unknown'),
                        'proof_time': metrics.get('proof_time_mean'),
                        'proof_size_kb': metrics.get('proof_size_kb'),
                        'circuit_size': metrics.get('circuit_size')
                    })
                    
                    completed += 1
                    print(f"✓ Completed: {config_name} (tolerance={tolerance})")
                    
                except Exception as e:
                    print(f"✗ Failed: {config_name} (tolerance={tolerance})")
                    print(f"  Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    self.experiment_log.append({
                        'experiment_id': f"scaling_{config_name}_tol{tolerance}",
                        'phase': 'scaling_study',
                        'gpu': 1,
                        'config_name': config_name,
                        'tolerance': tolerance,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        print(f"\n{'='*70}")
        print(f"GPU 1 Phase 2 Summary:")
        print(f"  Completed: {completed}")
        print(f"  Skipped: {skipped}")
        print(f"  Total processed: {exp_count}")
        print(f"{'='*70}\n")
        
        # Backup after completion
        self._backup_progress("gpu1_phase2_complete")


if __name__ == '__main__':
    print("="*70)
    print("GPU 1: PHASE 2 SCALING STUDY (Dense only)")
    print("="*70)
    print("\nGPU Assignment: CUDA_VISIBLE_DEVICES=1")
    print("Work Directory: temp_gpu1/")
    print("Log File: experiment_run_gpu1.log")
    print("\nExperiments to run:")
    print("  - Dense_Scaling_Small (tol=0.5)")
    print("  - Dense_Scaling_Small (tol=2.0)")
    print("  - Dense_Scaling_Large (tol=0.5)")
    print("  - Dense_Scaling_Large (tol=2.0)")
    print("="*70 + "\n")
    
    runner = Phase2Runner()
    runner.run_scaling_experiments()
    
    # Save final results
    log_data = runner.save_experiment_log()
    runner.print_summary()
    
    print("\n" + "="*70)
    print("GPU 1 PHASE 2 COMPLETE")
    print("="*70)
