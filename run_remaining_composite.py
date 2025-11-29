#!/usr/bin/env python
"""
Final Composite Experiments
Runs: CNN_ReLU (2), CNN_Poly (2) = 4 experiments
"""

import os
import sys
from pathlib import Path
import torch

# Use GPU 2
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.run_experiments import ExperimentRunner
from config.experiment_config import COMPOSITE_ARCHITECTURES, TOLERANCE_VALUES

# Run remaining architectures - try CNN_Poly first (CNN_ReLU has been OOM killed multiple times)
TARGET_ARCHS = ['CNN_Poly', 'CNN_ReLU']


class FinalCompositeRunner(ExperimentRunner):
    """Runner for final composite models"""
    
    def __init__(self):
        super().__init__()
        self.work_dir = Path('temp_final_composite')
        self.work_dir.mkdir(exist_ok=True)
        
    def run_composite_experiments(self):
        """Run CNN_ReLU and CNN_Poly experiments"""
        from src.models import create_composite_model
        from src.ezkl_utils import benchmark_model
        
        print(f"\n{'#'*70}")
        print(f"# FINAL COMPOSITE MODELS")
        print(f"# Target: CNN_ReLU (2), CNN_Poly (2)")
        print(f"# Work dir: {self.work_dir}")
        print(f"{'#'*70}\n")
        
        exp_count = 0
        completed = 0
        skipped = 0
        
        for arch_name in TARGET_ARCHS:
            if arch_name not in COMPOSITE_ARCHITECTURES:
                print(f"⚠️  {arch_name} not in config")
                continue
                
            arch_config = COMPOSITE_ARCHITECTURES[arch_name]
            
            # Find model checkpoint
            model_path = Path('models_pt') / 'composite' / f'{arch_name}_best.pt'
            if not model_path.exists():
                model_path = Path('models_pt') / 'composite' / f'{arch_name}_final.pt'
            
            if not model_path.exists():
                print(f"⚠️  Model not found: {model_path}")
                print(f"    Skipping {arch_name}")
                continue
            
            for tolerance in TOLERANCE_VALUES:
                exp_count += 1
                
                # Check if already completed
                result_file = self.results_dir / 'composite' / f'{arch_name}_tol{tolerance}_results.json'
                if result_file.exists():
                    print(f"⏭️  SKIPPED: {arch_name} (tol={tolerance}) - already completed")
                    skipped += 1
                    continue
                
                print(f"\n[Final-{exp_count}/4] Running: {arch_name} (tolerance={tolerance})\n")
                
                try:
                    # Load pre-trained model
                    model = create_composite_model(arch_name, arch_config)
                    checkpoint = torch.load(model_path, weights_only=False)
                    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                    model.load_state_dict(state_dict)
                    model.eval()
                    
                    print(f"  Loaded model from {model_path}")
                    
                    # Run benchmark
                    metrics = benchmark_model(
                        model=model,
                        model_name=arch_name,
                        input_shape=(1, 3, 32, 32),
                        tolerance=tolerance,
                        work_dir=self.work_dir / 'composite',
                        results_dir=self.results_dir / 'composite'
                    )
                    
                    # Log result
                    self.experiment_log.append({
                        'experiment_id': f"composite_{arch_name}_tol{tolerance}",
                        'phase': 'composite',
                        'architecture': arch_name,
                        'tolerance': tolerance,
                        'status': metrics.get('status', 'unknown'),
                        'proof_time': metrics.get('proof_time_mean'),
                        'proof_size_kb': metrics.get('proof_size_kb'),
                        'circuit_size': metrics.get('circuit_size')
                    })
                    
                    completed += 1
                    print(f"✓ Completed: {arch_name} (tolerance={tolerance})")
                    
                except Exception as e:
                    print(f"✗ Failed: {arch_name} (tolerance={tolerance})")
                    print(f"  Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    self.experiment_log.append({
                        'experiment_id': f"composite_{arch_name}_tol{tolerance}",
                        'phase': 'composite',
                        'architecture': arch_name,
                        'tolerance': tolerance,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        print(f"\n{'='*70}")
        print(f"Final Composite Summary:")
        print(f"  Completed: {completed}")
        print(f"  Skipped: {skipped}")
        print(f"  Total processed: {exp_count}")
        print(f"{'='*70}\n")
        
        # Backup after completion
        self._backup_progress("final_composite_complete")


if __name__ == '__main__':
    print("="*70)
    print("FINAL COMPOSITE EXPERIMENTS")
    print("="*70)
    print("\nGPU Assignment: CUDA_VISIBLE_DEVICES=1")
    print("Work Directory: temp_final_composite/")
    print("Log File: experiment_run_final_composite.log")
    print("\nExperiments to run:")
    print("  - CNN_ReLU (tol=0.5, tol=2.0)")
    print("  - CNN_Poly (tol=0.5, tol=2.0)")
    print("\nThis will complete all 26/26 feasible experiments!")
    print("="*70 + "\n")
    
    runner = FinalCompositeRunner()
    runner.run_composite_experiments()
    
    # Save final results
    log_data = runner.save_experiment_log()
    runner.print_summary()
    
    print("\n" + "="*70)
    print("🎉 ALL 26 EXPERIMENTS COMPLETE! 🎉")
    print("="*70)
