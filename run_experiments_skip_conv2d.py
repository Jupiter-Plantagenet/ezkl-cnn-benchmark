#!/usr/bin/env python
"""
Modified experiment runner that skips Conv2d experiments due to memory constraints.

This script runs the remaining 18 experiments:
- Phase 1: BatchNorm2d, LayerNorm, Dense (6 experiments)
- Phase 2: Dense scaling only (4 experiments) 
- Phase 3: All composite models (8 experiments)

Conv2d experiments are postponed for later execution with reduced input sizes
or on a system with more RAM.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.run_experiments import ExperimentRunner

# Layers to skip due to OOM issues
SKIP_LAYERS = [
    'Conv2d_k3_s1',
    'Conv2d_k3_s2', 
    'DepthwiseConv2d'
]

# Scaling configs to skip
SKIP_SCALING = [
    'Conv2d_Scaling_Small',
    'Conv2d_Scaling_Large'
]


class ModifiedExperimentRunner(ExperimentRunner):
    """Modified runner that skips Conv2d experiments"""
    
    def run_core_layer_experiments(self):
        """Run core layer experiments, skipping Conv2d variants"""
        from config.experiment_config import LAYER_CONFIGS, TOLERANCE_VALUES
        
        print(f"\n{'#'*70}")
        print(f"# PHASE 1: CORE LAYER EXPERIMENTS (Modified - Skipping Conv2d)")
        print(f"{'#'*70}\n")
        
        print("⚠️  SKIPPING Conv2d experiments due to memory constraints:")
        for layer in SKIP_LAYERS:
            print(f"   - {layer}")
        print()
        
        exp_count = 0
        
        for layer_name, config in LAYER_CONFIGS.items():
            # Skip Conv2d layers
            if layer_name in SKIP_LAYERS:
                print(f"⏸️  SKIPPED: {layer_name} (Conv2d - postponed)")
                continue
                
            for tolerance in TOLERANCE_VALUES:
                exp_count += 1
                print(f"\n[{exp_count}/18] Running: {layer_name} (tolerance={tolerance})\n")
                
                try:
                    # Import here to avoid circular dependency
                    from src.models import create_single_layer_model
                    from src.ezkl_utils import benchmark_model
                    
                    # Create model
                    model = create_single_layer_model(layer_name, config)
                    
                    # Run benchmark
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
                        'layer_name': layer_name,
                        'tolerance': tolerance,
                        'status': metrics.get('status', 'unknown'),
                        'proof_time': metrics.get('proof_time_mean'),
                        'proof_size_kb': metrics.get('proof_size_kb'),
                        'circuit_size': metrics.get('circuit_size')
                    })
                    
                    print(f"✓ Completed: {layer_name} (tolerance={tolerance})")
                    
                except Exception as e:
                    print(f"✗ Failed: {layer_name} (tolerance={tolerance})")
                    print(f"  Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    self.experiment_log.append({
                        'experiment_id': f"core_{layer_name}_tol{tolerance}",
                        'phase': 'core_layers',
                        'layer_name': layer_name,
                        'tolerance': tolerance,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        print(f"\n{'='*70}")
        print(f"Phase 1 completed: {exp_count} experiments (6 layers × 2 scales)")
        print(f"{'='*70}\n")
        
        # Backup after phase 1
        self._backup_progress("phase1_complete_no_conv2d")
    
    def run_scaling_experiments(self):
        """Run scaling study, skipping Conv2d variants"""
        from config.experiment_config import SCALING_CONFIGS, TOLERANCE_VALUES
        from src.models import create_single_layer_model
        from src.ezkl_utils import benchmark_model
        
        print(f"\n{'#'*70}")
        print(f"# PHASE 2: SCALING STUDY (Modified - Dense only)")
        print(f"{'#'*70}\n")
        
        print("⚠️  SKIPPING Conv2d scaling experiments:")
        for config in SKIP_SCALING:
            print(f"   - {config}")
        print()
        
        exp_count = 0
        
        for config_name, config_spec in SCALING_CONFIGS.items():
            # Skip Conv2d scaling experiments
            if config_name in SKIP_SCALING:
                print(f"⏸️  SKIPPED: {config_name} (Conv2d scaling - postponed)")
                continue
            
            for tolerance in TOLERANCE_VALUES:
                exp_count += 1
                print(f"\n[{exp_count}/4] Running: {config_name} (tolerance={tolerance})\n")
                
                try:
                    # Get base layer config
                    base_layer = config_spec['base_layer']
                    from config.experiment_config import LAYER_CONFIGS
                    base_config = LAYER_CONFIGS[base_layer]
                    
                    # Create modified config
                    modified_config = {
                        'type': base_config['type'],
                        'input_shape': config_spec['input_shape'],
                        'params': config_spec['params']
                    }
                    
                    # Create model
                    model = create_single_layer_model(base_layer, modified_config)
                    
                    # Run benchmark
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
                        'config_name': config_name,
                        'base_layer': base_layer,
                        'tolerance': tolerance,
                        'status': metrics.get('status', 'unknown'),
                        'proof_time': metrics.get('proof_time_mean'),
                        'proof_size_kb': metrics.get('proof_size_kb'),
                        'circuit_size': metrics.get('circuit_size')
                    })
                    
                    print(f"✓ Completed: {config_name} (tolerance={tolerance})")
                    
                except Exception as e:
                    print(f"✗ Failed: {config_name} (tolerance={tolerance})")
                    print(f"  Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    self.experiment_log.append({
                        'experiment_id': f"scaling_{config_name}_tol{tolerance}",
                        'phase': 'scaling_study',
                        'config_name': config_name,
                        'tolerance': tolerance,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        print(f"\n{'='*70}")
        print(f"Phase 2 completed: {exp_count} scaling experiments (Dense only)")
        print(f"{'='*70}\n")
        
        # Backup after phase 2
        self._backup_progress("phase2_complete_no_conv2d")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run EZKL CNN benchmarks (Skip Conv2d)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip CIFAR-10 training (load pre-trained models)')
    parser.add_argument('--phase', choices=['core', 'scaling', 'composite', 'all'],
                        default='all', help='Which phase to run')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MODIFIED EXPERIMENT RUNNER - SKIPPING Conv2d")
    print("="*70)
    print("\nPostponed experiments (will run later):")
    print("  - Conv2d_k3_s1 (2 experiments)")
    print("  - Conv2d_k3_s2 (2 experiments)")
    print("  - DepthwiseConv2d (2 experiments)")
    print("  - Conv2d_Scaling_Small (2 experiments)")
    print("  - Conv2d_Scaling_Large (2 experiments)")
    print("\nRunning experiments:")
    print("  - BatchNorm2d, LayerNorm, Dense (6 experiments)")
    print("  - Dense scaling studies (4 experiments)")
    print("  - All composite models (8 experiments)")
    print("  Total: 18 experiments")
    print("="*70 + "\n")
    
    runner = ModifiedExperimentRunner()
    
    if args.phase == 'all' or args.phase == 'core':
        runner.run_core_layer_experiments()
    
    if args.phase == 'all' or args.phase == 'scaling':
        runner.run_scaling_experiments()
    
    if args.phase == 'all' or args.phase == 'composite':
        runner.run_composite_experiments(skip_training=args.skip_training)
    
    # Save final results
    log_data = runner.save_experiment_log()
    runner.print_summary()
    
    print("\n" + "="*70)
    print("EXPERIMENT RUN COMPLETE")
    print("="*70)
    print("\nNote: Conv2d experiments postponed due to memory constraints.")
    print("See ISSUES_LOG.md for details.")
    print("\nTo run Conv2d experiments later:")
    print("  1. Reduce input sizes in config/experiment_config.py")
    print("  2. Or use a system with 256GB+ RAM")
    print("  3. Run: python run_conv2d_experiments.py")
    print("="*70)
