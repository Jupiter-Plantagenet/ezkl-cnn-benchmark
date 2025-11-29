#!/usr/bin/env python
"""
Comprehensive Analysis of EZKL CNN Benchmark Results
Analyzes all experimental results and generates insights
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ResultsAnalyzer:
    """Analyzer for EZKL benchmark results"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.data = []
        self.df = None
        
    def load_all_results(self):
        """Load all result JSON files"""
        print("Loading results...")
        
        for category in ['core_layers', 'scaling_study', 'composite']:
            category_dir = self.results_dir / category
            if not category_dir.exists():
                continue
                
            for result_file in category_dir.glob('*.json'):
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        data['category'] = category
                        data['filename'] = result_file.name
                        self.data.append(data)
                except Exception as e:
                    print(f"  Error loading {result_file}: {e}")
        
        print(f"  Loaded {len(self.data)} results")
        
        # Convert to DataFrame
        self.df = pd.DataFrame(self.data)
        
        # Tolerance should already be in the data
        if 'tolerance' not in self.df.columns:
            self.df['tolerance'] = self.df['model_name'].str.extract(r'tol([\d.]+)')[0].astype(float)
        
        # Clean model names (remove tolerance suffix)
        self.df['model_base'] = self.df['model_name'].str.replace(r'_tol[\d.]+', '', regex=True)
        
        return self.df
    
    def print_summary_statistics(self):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80 + "\n")
        
        print(f"Total experiments: {len(self.df)}")
        print(f"Categories: {self.df['category'].nunique()}")
        print(f"Unique models: {self.df['model_base'].nunique()}")
        print(f"Tolerances tested: {sorted(self.df['tolerance'].unique())}")
        
        print("\n" + "-"*80)
        print("By Category:")
        print("-"*80)
        for cat in ['core_layers', 'scaling_study', 'composite']:
            cat_df = self.df[self.df['category'] == cat]
            if len(cat_df) > 0:
                print(f"\n{cat.replace('_', ' ').title()}:")
                print(f"  Experiments: {len(cat_df)}")
                print(f"  Models: {cat_df['model_base'].nunique()}")
                print(f"  Avg proof time: {cat_df['proof_time_mean'].mean():.2f}s")
                print(f"  Avg proof size: {cat_df['proof_size_kb'].mean():.2f} KB")
                print(f"  Avg circuit size: {cat_df['circuit_size'].mean():.0f}")
        
        print("\n" + "-"*80)
        print("Overall Metrics:")
        print("-"*80)
        print(f"  Proof time:   {self.df['proof_time_mean'].mean():.2f}s ± {self.df['proof_time_mean'].std():.2f}s")
        print(f"                Min: {self.df['proof_time_mean'].min():.2f}s, Max: {self.df['proof_time_mean'].max():.2f}s")
        print(f"  Proof size:   {self.df['proof_size_kb'].mean():.2f} KB ± {self.df['proof_size_kb'].std():.2f} KB")
        print(f"                Min: {self.df['proof_size_kb'].min():.2f} KB, Max: {self.df['proof_size_kb'].max():.2f} KB")
        print(f"  Circuit size: {self.df['circuit_size'].mean():.0f} ± {self.df['circuit_size'].std():.0f}")
        print(f"                Min: {self.df['circuit_size'].min():.0f}, Max: {self.df['circuit_size'].max():.0f}")
        print(f"  Verify time:  {self.df['verify_time'].mean():.3f}s ± {self.df['verify_time'].std():.3f}s")
    
    def analyze_tolerance_impact(self):
        """Analyze impact of tolerance on proof metrics"""
        print("\n" + "="*80)
        print("TOLERANCE IMPACT ANALYSIS")
        print("="*80 + "\n")
        
        # Group by model and tolerance
        tolerance_comparison = self.df.groupby(['model_base', 'tolerance']).agg({
            'proof_time_mean': 'mean',
            'proof_size_kb': 'mean',
            'circuit_size': 'mean'
        }).reset_index()
        
        # Pivot to compare tolerances
        for metric in ['proof_time_mean', 'proof_size_kb', 'circuit_size']:
            pivot = tolerance_comparison.pivot(index='model_base', columns='tolerance', values=metric)
            if len(pivot.columns) == 2:
                pivot['ratio'] = pivot[2.0] / pivot[0.5]
                
        print("Average impact of tolerance (2.0 vs 0.5):")
        
        # Calculate overall ratios
        low_tol = self.df[self.df['tolerance'] == 0.5]
        high_tol = self.df[self.df['tolerance'] == 2.0]
        
        if len(low_tol) > 0 and len(high_tol) > 0:
            print(f"  Proof time ratio:   {high_tol['proof_time_mean'].mean() / low_tol['proof_time_mean'].mean():.3f}x")
            print(f"  Proof size ratio:   {high_tol['proof_size_kb'].mean() / low_tol['proof_size_kb'].mean():.3f}x")
            print(f"  Circuit size ratio: {high_tol['circuit_size'].mean() / low_tol['circuit_size'].mean():.3f}x")
            
            print("\n  Interpretation:")
            time_ratio = high_tol['proof_time_mean'].mean() / low_tol['proof_time_mean'].mean()
            if time_ratio < 1.1:
                print(f"    - Tolerance has MINIMAL impact on proof time (~{(time_ratio-1)*100:.1f}% increase)")
            else:
                print(f"    - Tolerance INCREASES proof time by ~{(time_ratio-1)*100:.1f}%")
    
    def analyze_layer_types(self):
        """Analyze different layer types"""
        print("\n" + "="*80)
        print("LAYER TYPE ANALYSIS (Core Layers)")
        print("="*80 + "\n")
        
        core_df = self.df[self.df['category'] == 'core_layers'].copy()
        
        if len(core_df) == 0:
            return
        
        # Categorize layers
        def categorize_layer(name):
            if any(x in name for x in ['ReLU', 'SiLU', 'Tanh', 'Poly']):
                return 'Activation'
            elif any(x in name for x in ['MaxPool', 'AvgPool']):
                return 'Pooling'
            elif 'BatchNorm' in name:
                return 'Normalization'
            elif 'Dense' in name:
                return 'Linear'
            return 'Other'
        
        core_df['layer_type'] = core_df['model_base'].apply(categorize_layer)
        
        # Group by layer type
        by_type = core_df.groupby('layer_type').agg({
            'proof_time_mean': ['mean', 'std', 'min', 'max'],
            'proof_size_kb': ['mean', 'std'],
            'circuit_size': ['mean', 'std']
        }).round(2)
        
        print("By Layer Type:")
        print(by_type)
        
        print("\n" + "-"*80)
        print("Fastest to Slowest (avg proof time):")
        print("-"*80)
        ranking = core_df.groupby('model_base')['proof_time_mean'].mean().sort_values()
        for i, (model, time) in enumerate(ranking.items(), 1):
            circuit_size = core_df[core_df['model_base'] == model]['circuit_size'].iloc[0]
            print(f"  {i:2d}. {model:20s} {time:8.3f}s  (circuit: {circuit_size:,})")
    
    def analyze_composite_models(self):
        """Analyze composite CNN models"""
        print("\n" + "="*80)
        print("COMPOSITE MODEL ANALYSIS")
        print("="*80 + "\n")
        
        comp_df = self.df[self.df['category'] == 'composite'].copy()
        
        if len(comp_df) == 0:
            print("No composite models found.")
            return
        
        print("Composite Models Performance:")
        print("-"*80)
        
        for model in comp_df['model_base'].unique():
            model_data = comp_df[comp_df['model_base'] == model]
            print(f"\n{model}:")
            print(f"  Experiments: {len(model_data)}")
            print(f"  Avg proof time: {model_data['proof_time_mean'].mean():.2f}s")
            print(f"  Avg proof size: {model_data['proof_size_kb'].mean():.2f} KB")
            print(f"  Circuit size: {model_data['circuit_size'].iloc[0]:,}")
            print(f"  Verification: {model_data['verify_time'].mean():.3f}s")
        
        print("\n" + "-"*80)
        print("Comparison to Simple Layers:")
        print("-"*80)
        
        core_avg = self.df[self.df['category'] == 'core_layers']['proof_time_mean'].mean()
        comp_avg = comp_df['proof_time_mean'].mean()
        
        print(f"  Simple layers avg: {core_avg:.2f}s")
        print(f"  Composite models avg: {comp_avg:.2f}s")
        print(f"  Ratio: {comp_avg/core_avg:.1f}x slower")
        print(f"  Interpretation: Composite models are {comp_avg/core_avg:.1f}x more expensive")
    
    def analyze_scaling(self):
        """Analyze scaling study results"""
        print("\n" + "="*80)
        print("SCALING STUDY ANALYSIS")
        print("="*80 + "\n")
        
        scale_df = self.df[self.df['category'] == 'scaling_study'].copy()
        
        if len(scale_df) == 0:
            print("No scaling study data found.")
            return
        
        print("Scaling Impact:")
        print("-"*80)
        
        for model in scale_df['model_base'].unique():
            model_data = scale_df[scale_df['model_base'] == model]
            print(f"\n{model}:")
            for _, row in model_data.iterrows():
                print(f"  Tolerance {row['tolerance']}: {row['proof_time_mean']:.2f}s, "
                      f"circuit: {row['circuit_size']:,}, size: {row['proof_size_kb']:.2f} KB")
        
        # Compare small vs large
        if 'Small' in scale_df['model_base'].values[0] or 'Large' in scale_df['model_base'].values[0]:
            small = scale_df[scale_df['model_base'].str.contains('Small')]
            large = scale_df[scale_df['model_base'].str.contains('Large')]
            
            if len(small) > 0 and len(large) > 0:
                print("\n" + "-"*80)
                print("Small vs Large:")
                print("-"*80)
                print(f"  Small avg proof time: {small['proof_time_mean'].mean():.2f}s")
                print(f"  Large avg proof time: {large['proof_time_mean'].mean():.2f}s")
                print(f"  Ratio: {large['proof_time_mean'].mean() / small['proof_time_mean'].mean():.2f}x")
    
    def generate_insights(self):
        """Generate key insights from the data"""
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80 + "\n")
        
        insights = []
        
        # 1. Most efficient operations
        fastest = self.df.nsmallest(3, 'proof_time_mean')
        insights.append(f"1. Fastest operations: {', '.join(fastest['model_base'].values)} "
                       f"({fastest['proof_time_mean'].mean():.2f}s avg)")
        
        # 2. Most expensive operations
        slowest = self.df.nlargest(3, 'proof_time_mean')
        insights.append(f"2. Slowest operations: {', '.join(slowest['model_base'].values)} "
                       f"({slowest['proof_time_mean'].mean():.2f}s avg)")
        
        # 3. Circuit size range
        insights.append(f"3. Circuit complexity ranges from {self.df['circuit_size'].min():,} to "
                       f"{self.df['circuit_size'].max():,} constraints ({self.df['circuit_size'].max() / self.df['circuit_size'].min():.0f}x difference)")
        
        # 4. Proof size consistency
        proof_size_cv = self.df['proof_size_kb'].std() / self.df['proof_size_kb'].mean()
        insights.append(f"4. Proof sizes are relatively consistent (CV: {proof_size_cv:.2%}), "
                       f"ranging {self.df['proof_size_kb'].min():.1f}-{self.df['proof_size_kb'].max():.1f} KB")
        
        # 5. Verification efficiency
        verify_avg = self.df['verify_time'].mean()
        insights.append(f"5. Verification is fast: {verify_avg:.3f}s average "
                       f"({self.df['verify_time'].max():.3f}s worst case)")
        
        for insight in insights:
            print(f"  {insight}\n")
        
        # Performance tiers
        print("-"*80)
        print("Performance Tiers (by proof time):")
        print("-"*80)
        
        self.df['tier'] = pd.cut(self.df['proof_time_mean'], 
                                  bins=[0, 10, 100, float('inf')],
                                  labels=['Fast (<10s)', 'Medium (10-100s)', 'Slow (>100s)'])
        
        for tier in ['Fast (<10s)', 'Medium (10-100s)', 'Slow (>100s)']:
            tier_df = self.df[self.df['tier'] == tier]
            if len(tier_df) > 0:
                print(f"\n  {tier}: {len(tier_df)} experiments")
                print(f"    Models: {', '.join(tier_df['model_base'].unique())}")
    
    def save_summary_csv(self, output_file='analysis/results_summary.csv'):
        """Save summary DataFrame to CSV"""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        summary_df = self.df[[
            'model_name', 'category', 'model_base', 'tolerance',
            'proof_time_mean', 'proof_time_std', 'proof_size_kb',
            'circuit_size', 'verify_time', 'status'
        ]].copy()
        
        summary_df.to_csv(output_path, index=False)
        print(f"\nSummary saved to: {output_path}")
        
        return summary_df


def main():
    """Main analysis routine"""
    print("="*80)
    print("EZKL CNN BENCHMARK - RESULTS ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer()
    
    # Load data
    df = analyzer.load_all_results()
    
    if df is None or len(df) == 0:
        print("No results found!")
        return
    
    # Run analyses
    analyzer.print_summary_statistics()
    analyzer.analyze_layer_types()
    analyzer.analyze_tolerance_impact()
    analyzer.analyze_scaling()
    analyzer.analyze_composite_models()
    analyzer.generate_insights()
    
    # Save summary
    analyzer.save_summary_csv()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review analysis output above")
    print("  2. Check analysis/results_summary.csv for detailed data")
    print("  3. Generate visualizations with plot_results.py")
    print("  4. Update ISSUES_LOG.md with final summary")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
