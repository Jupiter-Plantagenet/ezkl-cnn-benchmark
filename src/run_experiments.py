"""
Automated experiment runner for all 40 benchmark experiments.
Orchestrates:
  - 24 core layer experiments (12 layers × 2 tolerances)
  - 8 scaling study experiments (4 configs × 2 tolerances)
  - 8 composite architecture experiments (4 models × 2 tolerances)
"""

import torch
import json
import sys
from pathlib import Path
from datetime import datetime
import traceback

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_single_layer_model, create_composite_model
from src.ezkl_utils import benchmark_model
from src.train_cifar10 import train_composite_model
from src.backup_utils import backup_results
from config.experiment_config import (
    LAYER_CONFIGS,
    SCALING_CONFIGS,
    COMPOSITE_ARCHITECTURES,
    TOLERANCE_VALUES,
    TRAINING_CONFIG,
    EXPERIMENT_COUNTS
)


class ExperimentRunner:
    """Manages the execution of all benchmark experiments"""
    
    def __init__(self, results_dir='./results', work_dir='./temp', models_dir='./models_pt'):
        self.results_dir = Path(results_dir)
        self.work_dir = Path(work_dir)
        self.models_dir = Path(models_dir)
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        (self.results_dir / 'core_layers').mkdir(exist_ok=True)
        (self.results_dir / 'scaling_study').mkdir(exist_ok=True)
        (self.results_dir / 'composite').mkdir(exist_ok=True)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.experiment_log = []
        self.backup_count = 0
        
        print(f"\n{'='*70}")
        print(f"EZKL CNN Benchmark - Automated Experiment Runner")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Results directory: {self.results_dir}")
        print(f"Total experiments planned: {EXPERIMENT_COUNTS['total']}")
        print(f"  - Core layers: {EXPERIMENT_COUNTS['core_layers']}")
        print(f"  - Scaling study: {EXPERIMENT_COUNTS['scaling_study']}")
        print(f"  - Composite: {EXPERIMENT_COUNTS['composite']}")
        print(f"{'='*70}\n")
    
    def run_core_layer_experiments(self):
        """
        Run core layer experiments: 12 layers × 2 tolerances = 24 experiments
        """
        print(f"\n{'#'*70}")
        print(f"# PHASE 1: CORE LAYER EXPERIMENTS ({EXPERIMENT_COUNTS['core_layers']} experiments)")
        print(f"{'#'*70}\n")
        
        exp_count = 0
        total = EXPERIMENT_COUNTS['core_layers']
        
        for layer_name, config in LAYER_CONFIGS.items():
            for tolerance in TOLERANCE_VALUES:
                exp_count += 1
                print(f"\n[{exp_count}/{total}] Running: {layer_name} (tolerance={tolerance})")
                
                try:
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
        print(f"Phase 1 completed: {exp_count} core layer experiments")
        print(f"{'='*70}\n")
        
        # Backup after phase 1
        self._backup_progress("phase1_complete")
    
    def run_scaling_experiments(self):
        """
        Run scaling study: 4 configs × 2 tolerances = 8 experiments
        """
        print(f"\n{'#'*70}")
        print(f"# PHASE 2: SCALING STUDY ({EXPERIMENT_COUNTS['scaling_study']} experiments)")
        print(f"{'#'*70}\n")
        
        exp_count = 0
        total = EXPERIMENT_COUNTS['scaling_study']
        
        for config_name, config_spec in SCALING_CONFIGS.items():
            for tolerance in TOLERANCE_VALUES:
                exp_count += 1
                print(f"\n[{exp_count}/{total}] Running: {config_name} (tolerance={tolerance})")
                
                try:
                    # Get base layer type
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
        print(f"Phase 2 completed: {exp_count} scaling experiments")
        print(f"{'='*70}\n")
        
        # Backup after phase 2
        self._backup_progress("phase2_complete")
    
    def run_composite_experiments(self, skip_training=False):
        """
        Run composite architecture experiments: 4 models × 2 tolerances = 8 experiments
        
        Args:
            skip_training: If True, load pre-trained models instead of training
        """
        print(f"\n{'#'*70}")
        print(f"# PHASE 3: COMPOSITE ARCHITECTURE EXPERIMENTS ({EXPERIMENT_COUNTS['composite']} experiments)")
        print(f"{'#'*70}\n")
        
        # First, train all composite models if needed
        if not skip_training:
            print("\nTraining composite models on CIFAR-10...")
            for arch_name, arch_config in COMPOSITE_ARCHITECTURES.items():
                print(f"\nTraining {arch_name}...")
                model = create_composite_model(arch_name, arch_config)
                train_results = train_composite_model(
                    model=model,
                    architecture_name=arch_name,
                    config=TRAINING_CONFIG,
                    device=self.device,
                    save_dir=str(self.models_dir)
                )
                print(f"  Final accuracy: {train_results['final_test_acc']:.2f}%")
        else:
            print("\nSkipping training, will use pre-trained models")
        
        # Now run EZKL benchmarks
        print("\nRunning EZKL benchmarks on composite models...")
        
        exp_count = 0
        total = EXPERIMENT_COUNTS['composite']
        
        for arch_name, arch_config in COMPOSITE_ARCHITECTURES.items():
            for tolerance in TOLERANCE_VALUES:
                exp_count += 1
                print(f"\n[{exp_count}/{total}] Running: {arch_name} (tolerance={tolerance})")
                
                try:
                    # Load trained model
                    model = create_composite_model(arch_name, arch_config)
                    model_path = self.models_dir / 'composite' / f'{arch_name}_best.pt'
                    
                    if model_path.exists():
                        checkpoint = torch.load(model_path, map_location='cpu')
                        model.load_state_dict(checkpoint['model_state_dict'])
                        test_acc = checkpoint.get('test_acc', 0.0)
                        print(f"  Loaded model with test accuracy: {test_acc:.2f}%")
                    else:
                        print(f"  Warning: No trained model found at {model_path}, using untrained model")
                        test_acc = 0.0
                    
                    # Run benchmark
                    metrics = benchmark_model(
                        model=model,
                        model_name=arch_name,
                        input_shape=(1, 3, 32, 32),  # CIFAR-10 input
                        tolerance=tolerance,
                        work_dir=self.work_dir / 'composite',
                        results_dir=self.results_dir / 'composite'
                    )
                    
                    # Add accuracy to metrics
                    metrics['test_accuracy'] = test_acc
                    
                    # Log result
                    self.experiment_log.append({
                        'experiment_id': f"composite_{arch_name}_tol{tolerance}",
                        'phase': 'composite',
                        'architecture': arch_name,
                        'tolerance': tolerance,
                        'test_accuracy': test_acc,
                        'status': metrics.get('status', 'unknown'),
                        'proof_time': metrics.get('proof_time_mean'),
                        'proof_size_kb': metrics.get('proof_size_kb'),
                        'circuit_size': metrics.get('circuit_size')
                    })
                    
                    print(f"✓ Completed: {arch_name} (tolerance={tolerance})")
                    
                except Exception as e:
                    print(f"✗ Failed: {arch_name} (tolerance={tolerance})")
                    print(f"  Error: {str(e)}")
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
        print(f"Phase 3 completed: {exp_count} composite experiments")
        print(f"{'='*70}\n")
        
        # Backup after phase 3
        self._backup_progress("phase3_complete")
    
    def _backup_progress(self, checkpoint_name):
        """Create a backup of current progress"""
        try:
            self.backup_count += 1
            print(f"\n📦 Creating backup checkpoint: {checkpoint_name}...")
            backup_path = backup_results(backup_dir='./backups')
            print(f"✓ Backup #{self.backup_count} completed: {backup_path.name}\n")
        except Exception as e:
            print(f"⚠ Backup failed (non-critical): {e}\n")
    
    def save_experiment_log(self):
        """Save complete experiment log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.results_dir / f"experiment_log_{timestamp}.json"
        
        log_data = {
            'timestamp': timestamp,
            'device': self.device,
            'total_experiments': len(self.experiment_log),
            'successful': sum(1 for e in self.experiment_log if e.get('status') == 'success'),
            'failed': sum(1 for e in self.experiment_log if e.get('status') == 'failed'),
            'experiments': self.experiment_log
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\nExperiment log saved to: {log_path}")
        
        return log_data
    
    def print_summary(self):
        """Print experiment summary"""
        successful = sum(1 for e in self.experiment_log if e.get('status') == 'success')
        failed = sum(1 for e in self.experiment_log if e.get('status') == 'failed')
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*70}")
        print(f"Total experiments: {len(self.experiment_log)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {100*successful/len(self.experiment_log):.1f}%")
        print(f"{'='*70}\n")
    
    def run_all_experiments(self, skip_training=False):
        """Run all 40 experiments"""
        start_time = datetime.now()
        
        print(f"\nStarting all experiments at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all phases
        self.run_core_layer_experiments()
        self.run_scaling_experiments()
        self.run_composite_experiments(skip_training=skip_training)
        
        # Save log
        self.save_experiment_log()
        
        # Print summary
        self.print_summary()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\nCompleted all experiments at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run EZKL CNN benchmark experiments')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training composite models (use pre-trained)')
    parser.add_argument('--core-only', action='store_true',
                        help='Run only core layer experiments')
    parser.add_argument('--scaling-only', action='store_true',
                        help='Run only scaling experiments')
    parser.add_argument('--composite-only', action='store_true',
                        help='Run only composite experiments')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ExperimentRunner()
    
    # Run requested experiments
    if args.core_only:
        runner.run_core_layer_experiments()
    elif args.scaling_only:
        runner.run_scaling_experiments()
    elif args.composite_only:
        runner.run_composite_experiments(skip_training=args.skip_training)
    else:
        runner.run_all_experiments(skip_training=args.skip_training)
    
    # Save and print summary
    runner.save_experiment_log()
    runner.print_summary()


if __name__ == '__main__':
    main()
