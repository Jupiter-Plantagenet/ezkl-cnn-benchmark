"""
Results analysis and aggregation pipeline.
Processes benchmark results and generates tables for the paper.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class ResultsAnalyzer:
    """Analyzes and aggregates benchmark results"""
    
    def __init__(self, results_dir='./results'):
        self.results_dir = Path(results_dir)
        self.core_results = []
        self.scaling_results = []
        self.composite_results = []
    
    def load_all_results(self):
        """Load all result JSON files"""
        print("Loading results...")
        
        # Load core layer results
        core_dir = self.results_dir / 'core_layers'
        if core_dir.exists():
            for result_file in core_dir.glob('*.json'):
                with open(result_file) as f:
                    data = json.load(f)
                    data['result_file'] = result_file.name
                    self.core_results.append(data)
        
        # Load scaling results
        scaling_dir = self.results_dir / 'scaling_study'
        if scaling_dir.exists():
            for result_file in scaling_dir.glob('*.json'):
                with open(result_file) as f:
                    data = json.load(f)
                    data['result_file'] = result_file.name
                    self.scaling_results.append(data)
        
        # Load composite results
        composite_dir = self.results_dir / 'composite'
        if composite_dir.exists():
            for result_file in composite_dir.glob('*.json'):
                with open(result_file) as f:
                    data = json.load(f)
                    data['result_file'] = result_file.name
                    self.composite_results.append(data)
        
        print(f"  Core layer results: {len(self.core_results)}")
        print(f"  Scaling results: {len(self.scaling_results)}")
        print(f"  Composite results: {len(self.composite_results)}")
    
    def create_core_layers_table(self):
        """Create summary table for core layer experiments"""
        if not self.core_results:
            print("No core layer results to analyze")
            return None
        
        df = pd.DataFrame(self.core_results)
        
        # Group by model and tolerance
        summary = df.groupby(['model_name', 'tolerance']).agg({
            'proof_time_mean': 'mean',
            'proof_size_kb': 'mean',
            'num_constraints': 'mean',
            'peak_memory_mean_gb': 'mean',
            'status': lambda x: (x == 'success').sum()
        }).reset_index()
        
        summary.columns = [
            'Layer', 'Tolerance', 'Proof Time (s)', 
            'Proof Size (KB)', 'Constraints', 'Peak Memory (GB)', 'Success Count'
        ]
        
        return summary
    
    def create_tolerance_comparison_table(self):
        """Create table comparing tolerance settings"""
        if not self.core_results:
            return None
        
        df = pd.DataFrame(self.core_results)
        df = df[df['status'] == 'success']
        
        # For each layer, compare tolerances
        pivot = df.pivot_table(
            index='model_name',
            columns='tolerance',
            values=['proof_time_mean', 'proof_size_kb', 'num_constraints'],
            aggfunc='mean'
        )
        
        return pivot
    
    def create_scaling_analysis(self):
        """Analyze scaling behavior"""
        if not self.scaling_results:
            return None
        
        df = pd.DataFrame(self.scaling_results)
        df = df[df['status'] == 'success']
        
        # Separate by base layer type
        conv_scaling = df[df['model_name'].str.contains('Conv2d')]
        dense_scaling = df[df['model_name'].str.contains('Dense')]
        
        return {
            'conv_scaling': conv_scaling,
            'dense_scaling': dense_scaling
        }
    
    def create_composite_comparison_table(self):
        """Create table comparing composite architectures"""
        if not self.composite_results:
            return None
        
        df = pd.DataFrame(self.composite_results)
        df = df[df['status'] == 'success']
        
        # Group by architecture and tolerance
        summary = df.groupby(['model_name', 'tolerance']).agg({
            'test_accuracy': 'first',
            'proof_time_mean': 'mean',
            'proof_size_kb': 'mean',
            'num_constraints': 'mean',
            'peak_memory_mean_gb': 'mean'
        }).reset_index()
        
        summary.columns = [
            'Architecture', 'Tolerance', 'Test Accuracy (%)',
            'Proof Time (s)', 'Proof Size (KB)', 'Constraints', 'Peak Memory (GB)'
        ]
        
        return summary
    
    def generate_latex_tables(self, output_dir='./paper'):
        """Generate LaTeX tables for the paper"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating LaTeX tables...")
        
        # Core layers table
        core_table = self.create_core_layers_table()
        if core_table is not None:
            latex_path = output_dir / 'table_core_layers.tex'
            with open(latex_path, 'w') as f:
                f.write(core_table.to_latex(index=False, float_format='%.2f'))
            print(f"  Saved core layers table to {latex_path}")
        
        # Tolerance comparison
        tolerance_table = self.create_tolerance_comparison_table()
        if tolerance_table is not None:
            latex_path = output_dir / 'table_tolerance_comparison.tex'
            with open(latex_path, 'w') as f:
                f.write(tolerance_table.to_latex(float_format='%.2f'))
            print(f"  Saved tolerance comparison to {latex_path}")
        
        # Composite comparison
        composite_table = self.create_composite_comparison_table()
        if composite_table is not None:
            latex_path = output_dir / 'table_composite_comparison.tex'
            with open(latex_path, 'w') as f:
                f.write(composite_table.to_latex(index=False, float_format='%.2f'))
            print(f"  Saved composite comparison to {latex_path}")
    
    def plot_pareto_frontier(self, output_dir='./paper'):
        """Generate Pareto frontier plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.composite_results:
            print("No composite results for Pareto plots")
            return
        
        df = pd.DataFrame(self.composite_results)
        df = df[df['status'] == 'success']
        
        # Create 2D Pareto plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Proof time vs Proof size
        for tolerance in df['tolerance'].unique():
            subset = df[df['tolerance'] == tolerance]
            axes[0].scatter(
                subset['proof_time_mean'],
                subset['proof_size_kb'],
                label=f'Tolerance {tolerance}',
                s=100,
                alpha=0.7
            )
            for _, row in subset.iterrows():
                axes[0].annotate(
                    row['model_name'],
                    (row['proof_time_mean'], row['proof_size_kb']),
                    fontsize=8
                )
        
        axes[0].set_xlabel('Proof Time (s)')
        axes[0].set_ylabel('Proof Size (KB)')
        axes[0].set_title('Latency vs Bandwidth')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Proof time vs Constraints
        for tolerance in df['tolerance'].unique():
            subset = df[df['tolerance'] == tolerance]
            axes[1].scatter(
                subset['proof_time_mean'],
                subset['num_constraints'],
                label=f'Tolerance {tolerance}',
                s=100,
                alpha=0.7
            )
            for _, row in subset.iterrows():
                axes[1].annotate(
                    row['model_name'],
                    (row['proof_time_mean'], row['num_constraints']),
                    fontsize=8
                )
        
        axes[1].set_xlabel('Proof Time (s)')
        axes[1].set_ylabel('Circuit Constraints')
        axes[1].set_title('Latency vs Circuit Complexity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Proof size vs Constraints
        for tolerance in df['tolerance'].unique():
            subset = df[df['tolerance'] == tolerance]
            axes[2].scatter(
                subset['proof_size_kb'],
                subset['num_constraints'],
                label=f'Tolerance {tolerance}',
                s=100,
                alpha=0.7
            )
            for _, row in subset.iterrows():
                axes[2].annotate(
                    row['model_name'],
                    (row['proof_size_kb'], row['num_constraints']),
                    fontsize=8
                )
        
        axes[2].set_xlabel('Proof Size (KB)')
        axes[2].set_ylabel('Circuit Constraints')
        axes[2].set_title('Bandwidth vs Circuit Complexity')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'pareto_frontiers.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n  Saved Pareto frontier plots to {plot_path}")
        plt.close()
    
    def generate_summary_report(self, output_path='./results/summary_report.txt'):
        """Generate a text summary report"""
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EZKL CNN BENCHMARK - SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Core layers summary
            if self.core_results:
                f.write("CORE LAYER EXPERIMENTS\n")
                f.write("-"*70 + "\n")
                df = pd.DataFrame(self.core_results)
                df_success = df[df['status'] == 'success']
                
                f.write(f"Total experiments: {len(df)}\n")
                f.write(f"Successful: {len(df_success)}\n\n")
                
                if len(df_success) > 0:
                    f.write("Average metrics (successful experiments):\n")
                    f.write(f"  Proof time: {df_success['proof_time_mean'].mean():.2f}s\n")
                    f.write(f"  Proof size: {df_success['proof_size_kb'].mean():.2f} KB\n")
                    f.write(f"  Circuit constraints: {df_success['num_constraints'].mean():.0f}\n")
                    f.write(f"  Peak memory: {df_success['peak_memory_mean_gb'].mean():.2f} GB\n\n")
            
            # Composite summary
            if self.composite_results:
                f.write("\nCOMPOSITE ARCHITECTURE EXPERIMENTS\n")
                f.write("-"*70 + "\n")
                df = pd.DataFrame(self.composite_results)
                df_success = df[df['status'] == 'success']
                
                f.write(f"Total experiments: {len(df)}\n")
                f.write(f"Successful: {len(df_success)}\n\n")
                
                if len(df_success) > 0:
                    for arch in df_success['model_name'].unique():
                        arch_data = df_success[df_success['model_name'] == arch]
                        f.write(f"\n{arch}:\n")
                        f.write(f"  Test accuracy: {arch_data['test_accuracy'].mean():.2f}%\n")
                        f.write(f"  Proof time: {arch_data['proof_time_mean'].mean():.2f}s\n")
                        f.write(f"  Proof size: {arch_data['proof_size_kb'].mean():.2f} KB\n")
                        f.write(f"  Constraints: {arch_data['num_constraints'].mean():.0f}\n")
        
        print(f"\n  Saved summary report to {output_path}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*70)
        print("RESULTS ANALYSIS")
        print("="*70)
        
        self.load_all_results()
        self.generate_latex_tables()
        self.plot_pareto_frontier()
        self.generate_summary_report()
        
        print("\n" + "="*70)
        print("Analysis complete!")
        print("="*70 + "\n")


def main():
    """Main entry point"""
    analyzer = ResultsAnalyzer()
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
