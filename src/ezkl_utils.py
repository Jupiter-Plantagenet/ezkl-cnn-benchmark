"""
EZKL integration utilities for circuit generation, proving, and metrics collection.
"""

import ezkl
import torch
import time
import os
import json
import psutil
import subprocess
from pathlib import Path
import numpy as np


class EZKLBenchmark:
    """Handles EZKL circuit generation, proving, and metrics collection"""
    
    def __init__(self, model, model_name, tolerance, work_dir='./temp'):
        """
        Initialize EZKL benchmark for a model.
        
        Args:
            model: PyTorch model
            model_name: Name identifier for the model
            tolerance: EZKL tolerance parameter
            work_dir: Working directory for EZKL artifacts
        """
        self.model = model
        self.model_name = model_name
        self.tolerance = tolerance
        self.work_dir = Path(work_dir) / f"{model_name}_tol{tolerance}"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.onnx_path = self.work_dir / "model.onnx"
        self.input_path = self.work_dir / "input.json"
        self.settings_path = self.work_dir / "settings.json"
        self.compiled_path = self.work_dir / "model.ezkl"
        self.pk_path = self.work_dir / "pk.key"
        self.vk_path = self.work_dir / "vk.key"
        self.srs_path = self.work_dir / "srs.bin"
        self.witness_path = self.work_dir / "witness.json"
        self.proof_path = self.work_dir / "proof.pf"
        
        self.metrics = {}
    
    def export_onnx(self, dummy_input):
        """Export PyTorch model to ONNX"""
        print(f"  Exporting to ONNX...")
        self.model.eval()
        torch.onnx.export(
            self.model,
            dummy_input,
            str(self.onnx_path),
            input_names=['input'],
            output_names=['output'],
            opset_version=14,
            do_constant_folding=True
        )
        
        # Save dummy input as JSON for EZKL - format per docs
        input_data = dummy_input.detach().numpy().reshape(-1).tolist()
        with open(self.input_path, 'w') as f:
            json.dump({"input_data": [input_data]}, f)
        
        return str(self.onnx_path)
    
    def generate_settings(self):
        """
        Generate EZKL settings with specified scale configuration.
        
        Note: The 'tolerance' parameter is our experimental variable that maps to
        EZKL's input_scale/param_scale. EZKL v23.0.3 doesn't have a 'tolerance' 
        attribute - it uses 'scale' to control fixed-point precision.
        """
        print(f"  Generating settings (experimental config={self.tolerance})...")
        
        # Map our experimental parameter to EZKL scale
        # EZKL scale controls fixed-point precision: higher = more accurate, larger circuit
        # Our "tolerance" 0.5 -> scale 10 (accuracy mode)
        # Our "tolerance" 2.0 -> scale 7 (efficiency mode)
        if self.tolerance <= 0.5:
            scale = 10  # High precision
        elif self.tolerance <= 1.0:
            scale = 8   # Medium precision
        else:
            scale = 7   # Low precision (smaller/faster)
        
        print(f"  Using EZKL scale={scale} (bits of precision)")
        
        # Create run args
        run_args = ezkl.PyRunArgs()
        run_args.input_scale = scale
        run_args.param_scale = scale
        run_args.input_visibility = "public"
        run_args.output_visibility = "public"
        run_args.param_visibility = "private"
        
        # Generate initial settings
        ezkl.gen_settings(
            model=str(self.onnx_path),
            output=str(self.settings_path),
            py_run_args=run_args
        )
        
        # Store experimental config and actual EZKL scale in metrics
        self.metrics['experimental_config'] = self.tolerance  # Our experimental variable
        self.metrics['ezkl_input_scale'] = scale  # Actual EZKL parameter used
        self.metrics['ezkl_param_scale'] = scale  # Actual EZKL parameter used
        
        return str(self.settings_path)
    
    def calibrate_settings(self):
        """Calibrate settings for optimal circuit size"""
        print(f"  Calibrating settings...")
        
        try:
            ezkl.calibrate_settings(
                data=str(self.input_path),
                model=str(self.onnx_path),
                settings=str(self.settings_path)
            )
            print(f"  Calibration successful")
        except Exception as e:
            # Calibration can fail for some models, continue without it
            print(f"  Calibration skipped (not critical): {str(e)[:100]}")
            pass
    
    def compile_circuit(self):
        """Compile the circuit"""
        print(f"  Compiling circuit...")
        start_time = time.time()
        
        ezkl.compile_circuit(
            model=str(self.onnx_path),
            compiled_circuit=str(self.compiled_path),
            settings_path=str(self.settings_path)
        )
        
        compile_time = time.time() - start_time
        self.metrics['compile_time'] = compile_time
        
        # Extract circuit complexity from settings
        with open(self.settings_path, 'r') as f:
            settings = json.load(f)
            self.metrics['circuit_size'] = settings.get('num_rows', 0)
            self.metrics['num_constraints'] = settings.get('total_const_size', 0)
    
    def setup_srs(self):
        """Setup structured reference string"""
        print(f"  Setting up SRS...")
        
        # Get logrows from settings
        with open(self.settings_path, 'r') as f:
            settings = json.load(f)
            logrows = settings.get('run_args', {}).get('logrows', 17)
        
        print(f"  Generating SRS with logrows={logrows}...")
        
        # Generate SRS locally (for testing) instead of downloading
        ezkl.gen_srs(
            srs_path=str(self.srs_path),
            logrows=logrows
        )
    
    def setup_keys(self):
        """Generate proving and verifying keys"""
        print(f"  Generating proving and verifying keys...")
        start_time = time.time()
        
        ezkl.setup(
            model=str(self.compiled_path),
            vk_path=str(self.vk_path),
            pk_path=str(self.pk_path),
            srs_path=str(self.srs_path)
        )
        
        setup_time = time.time() - start_time
        self.metrics['setup_time'] = setup_time
        
        # Get key sizes
        self.metrics['pk_size_kb'] = os.path.getsize(self.pk_path) / 1024
        self.metrics['vk_size_kb'] = os.path.getsize(self.vk_path) / 1024
    
    def generate_witness(self):
        """Generate witness from input"""
        print(f"  Generating witness...")
        
        ezkl.gen_witness(
            data=str(self.input_path),
            model=str(self.compiled_path),
            output=str(self.witness_path),
            vk_path=str(self.vk_path),
            srs_path=str(self.srs_path)
        )
    
    def generate_proof(self, num_runs=3):
        """
        Generate proof and measure performance.
        
        Args:
            num_runs: Number of runs to average timing
        
        Returns:
            Dictionary with proof metrics
        """
        print(f"  Generating proof ({num_runs} runs)...")
        
        proof_times = []
        peak_memories = []
        
        for run in range(num_runs):
            # Measure memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 ** 3)  # GB
            
            # Time proof generation
            start_time = time.time()
            
            ezkl.prove(
                witness=str(self.witness_path),
                model=str(self.compiled_path),
                pk_path=str(self.pk_path),
                proof_path=str(self.proof_path),
                srs_path=str(self.srs_path)
            )
            
            proof_time = time.time() - start_time
            proof_times.append(proof_time)
            
            # Measure memory after
            mem_after = process.memory_info().rss / (1024 ** 3)  # GB
            peak_memory = mem_after - mem_before
            peak_memories.append(peak_memory)
            
            print(f"    Run {run+1}: {proof_time:.2f}s, Memory delta: {peak_memory:.2f} GB")
        
        # Get proof size
        proof_size_kb = os.path.getsize(self.proof_path) / 1024
        
        # Store metrics
        self.metrics['proof_time_mean'] = np.mean(proof_times)
        self.metrics['proof_time_std'] = np.std(proof_times)
        self.metrics['proof_time_min'] = np.min(proof_times)
        self.metrics['proof_time_max'] = np.max(proof_times)
        self.metrics['proof_size_kb'] = proof_size_kb
        self.metrics['peak_memory_mean_gb'] = np.mean(peak_memories)
        self.metrics['peak_memory_std_gb'] = np.std(peak_memories)
        
        print(f"  Proof time: {self.metrics['proof_time_mean']:.2f}s ± {self.metrics['proof_time_std']:.2f}s")
        print(f"  Proof size: {proof_size_kb:.2f} KB")
        
        return self.metrics
    
    def verify_proof(self):
        """Verify the generated proof"""
        print(f"  Verifying proof...")
        start_time = time.time()
        
        is_valid = ezkl.verify(
            proof_path=str(self.proof_path),
            settings_path=str(self.settings_path),
            vk_path=str(self.vk_path),
            srs_path=str(self.srs_path)
        )
        
        verify_time = time.time() - start_time
        self.metrics['verify_time'] = verify_time
        self.metrics['proof_valid'] = is_valid
        
        print(f"  Verification: {'PASSED' if is_valid else 'FAILED'} ({verify_time:.4f}s)")
        
        return is_valid
    
    def run_full_benchmark(self, dummy_input, num_proof_runs=3):
        """
        Run complete EZKL benchmark pipeline.
        
        Args:
            dummy_input: Input tensor for the model
            num_proof_runs: Number of proof generation runs
        
        Returns:
            Dictionary with all metrics
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {self.model_name} (tolerance={self.tolerance})")
        print(f"{'='*60}")
        
        try:
            # Full pipeline
            self.export_onnx(dummy_input)
            self.generate_settings()
            self.calibrate_settings()
            self.compile_circuit()
            self.setup_srs()
            self.setup_keys()
            self.generate_witness()
            self.generate_proof(num_runs=num_proof_runs)
            self.verify_proof()
            
            # Add metadata
            self.metrics['model_name'] = self.model_name
            self.metrics['tolerance'] = self.tolerance
            self.metrics['status'] = 'success'
            
            print(f"\n✓ Benchmark completed successfully")
            
        except Exception as e:
            print(f"\n✗ Benchmark failed: {str(e)}")
            self.metrics['status'] = 'failed'
            self.metrics['error'] = str(e)
        
        return self.metrics
    
    def save_metrics(self, output_path):
        """Save metrics to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"  Metrics saved to {output_path}")


def benchmark_model(model, model_name, input_shape, tolerance, work_dir='./temp', results_dir='./results'):
    """
    Convenience function to benchmark a single model.
    
    Args:
        model: PyTorch model
        model_name: Name identifier
        input_shape: Shape of input tensor
        tolerance: EZKL tolerance parameter
        work_dir: Working directory for EZKL artifacts
        results_dir: Directory to save results
    
    Returns:
        Dictionary with benchmark metrics
    """
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Run benchmark
    benchmark = EZKLBenchmark(model, model_name, tolerance, work_dir)
    metrics = benchmark.run_full_benchmark(dummy_input)
    
    # Save results
    results_path = Path(results_dir) / f"{model_name}_tol{tolerance}_results.json"
    benchmark.save_metrics(results_path)
    
    return metrics


if __name__ == '__main__':
    # Test benchmark on a simple model
    import sys
    sys.path.append('..')
    from models import create_single_layer_model
    from config.experiment_config import LAYER_CONFIGS, TOLERANCE_VALUES
    
    # Test on ReLU
    layer_name = 'ReLU'
    config = LAYER_CONFIGS[layer_name]
    model = create_single_layer_model(layer_name, config)
    
    print(f"Testing benchmark on {layer_name}...")
    for tolerance in TOLERANCE_VALUES:
        metrics = benchmark_model(
            model,
            layer_name,
            config['input_shape'],
            tolerance
        )
        print(f"\nResults for tolerance={tolerance}:")
        print(json.dumps(metrics, indent=2))
