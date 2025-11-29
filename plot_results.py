#!/usr/bin/env python
"""
Generate visualizations for EZKL CNN Benchmark results
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality figures (IEEE conference format)
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14  # Increased for legibility
plt.rcParams['axes.labelsize'] = 16  # Larger axis labels
plt.rcParams['axes.titlesize'] = 18  # Not used (titles in captions)
plt.rcParams['xtick.labelsize'] = 13  # Larger tick labels
plt.rcParams['ytick.labelsize'] = 13  # Larger tick labels
plt.rcParams['legend.fontsize'] = 12  # Larger legend
plt.rcParams['figure.figsize'] = (8, 5)  # Better aspect ratio for IEEE column


def load_results(results_dir='results'):
    """Load all results into DataFrame"""
    data = []
    results_path = Path(results_dir)
    
    for category in ['core_layers', 'scaling_study', 'composite']:
        category_dir = results_path / category
        if not category_dir.exists():
            continue
            
        for result_file in category_dir.glob('*.json'):
            with open(result_file, 'r') as f:
                result = json.load(f)
                result['category'] = category
                data.append(result)
    
    df = pd.DataFrame(data)
    df['model_base'] = df['model_name'].str.replace(r'_tol[\d.]+', '', regex=True)
    
    return df


def plot_tolerance_comparison(df, output_dir='paper'):
    """Plot tolerance impact - expected vs actual"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['proof_time_mean', 'proof_size_kb', 'circuit_size']
    titles = ['Proof Time (s)', 'Proof Size (KB)', 'Circuit Size']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        # Get data for both tolerances
        low_tol = df[df['tolerance'] == 0.5][metric]
        high_tol = df[df['tolerance'] == 2.0][metric]
        
        if len(low_tol) > 0 and len(high_tol) > 0:
            ratio = high_tol.mean() / low_tol.mean()
            
            # Bar plot
            x = ['Tolerance 0.5\n(High Precision)', 'Tolerance 2.0\n(Low Precision)']
            y = [low_tol.mean(), high_tol.mean()]
            colors = ['#2ecc71', '#e74c3c']
            
            bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontweight='bold')
            
            # Add ratio annotation
            ax.text(0.5, max(y) * 0.9, f'Ratio: {ratio:.3f}x',
                   ha='center', va='top', transform=ax.transData,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   fontsize=11, fontweight='bold')
            
            ax.set_ylabel(title, fontweight='bold', fontsize=16)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add "Expected" vs "Actual" annotation for proof time
            if idx == 0:
                ax.text(0.5, -0.25, '❌ Expected: 2-10x difference\n✅ Actual: ~1.0x (negligible!)',
                       transform=ax.transAxes, ha='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                       fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tolerance_comparison.png', bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/tolerance_comparison.png")
    plt.close()


def plot_performance_tiers(df, output_dir='paper'):
    """Plot performance tier breakdown"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define tiers
    df['tier'] = pd.cut(df['proof_time_mean'], 
                        bins=[0, 10, 100, float('inf')],
                        labels=['Fast\n(<10s)', 'Medium\n(10-100s)', 'Slow\n(>100s)'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Tier distribution
    tier_counts = df['tier'].value_counts().sort_index()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    bars = ax1.bar(tier_counts.index, tier_counts.values, color=colors, 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax1.set_ylabel('Number of Experiments', fontweight='bold', fontsize=16)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Box plot by tier
    tier_data = [df[df['tier'] == tier]['proof_time_mean'].values 
                 for tier in ['Fast\n(<10s)', 'Medium\n(10-100s)', 'Slow\n(>100s)']]
    
    bp = ax2.boxplot(tier_data, labels=['Fast', 'Medium', 'Slow'],
                     patch_artist=True, notch=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Proof Time (seconds)', fontweight='bold', fontsize=16)
    ax2.set_xlabel('Performance Tier', fontweight='bold', fontsize=16)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_tiers.png', bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/performance_tiers.png")
    plt.close()


def plot_layer_comparison(df, output_dir='paper'):
    """Plot layer type comparison"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    core_df = df[df['category'] == 'core_layers'].copy()
    
    # Average by model base
    layer_stats = core_df.groupby('model_base').agg({
        'proof_time_mean': 'mean',
        'circuit_size': 'first'
    }).reset_index()
    
    layer_stats = layer_stats.sort_values('proof_time_mean')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by time
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(layer_stats)))
    
    bars = ax.barh(layer_stats['model_base'], layer_stats['proof_time_mean'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (bar, circuit) in enumerate(zip(bars, layer_stats['circuit_size'])):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f' {width:.1f}s ({circuit:,} constraints)',
               ha='left', va='center', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Average Proof Time (seconds)', fontweight='bold', fontsize=16)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, layer_stats['proof_time_mean'].max() * 1.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/layer_comparison.png', bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/layer_comparison.png")
    plt.close()


def plot_scaling_curves(df, output_dir='paper'):
    """Plot scaling behavior"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Get all experiments sorted by circuit size
    plot_df = df.copy()
    plot_df = plot_df.sort_values('circuit_size')
    
    # Categorize
    def categorize(name):
        if any(x in name for x in ['ReLU', 'SiLU', 'Tanh', 'Poly']):
            return 'Activation'
        elif any(x in name for x in ['Pool']):
            return 'Pooling'
        elif 'BatchNorm' in name:
            return 'Normalization'
        elif 'CNN' in name:
            return 'Composite'
        elif 'Dense' in name:
            return 'Linear'
        return 'Other'
    
    plot_df['type'] = plot_df['model_base'].apply(categorize)
    
    # Plot with different markers per type
    type_styles = {
        'Activation': ('o', '#3498db'),
        'Pooling': ('s', '#e74c3c'),
        'Normalization': ('^', '#f39c12'),
        'Linear': ('D', '#2ecc71'),
        'Composite': ('*', '#9b59b6')
    }
    
    for type_name, (marker, color) in type_styles.items():
        type_data = plot_df[plot_df['type'] == type_name]
        if len(type_data) > 0:
            ax.scatter(type_data['circuit_size'], type_data['proof_time_mean'],
                      s=100, marker=marker, c=color, alpha=0.7, 
                      edgecolors='black', linewidth=1, label=type_name)
    
    # Fit power law
    x = plot_df['circuit_size'].values
    y = plot_df['proof_time_mean'].values
    
    # Log-log fit
    log_x = np.log10(x)
    log_y = np.log10(y)
    coeffs = np.polyfit(log_x, log_y, 1)
    
    x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    y_fit = 10**(coeffs[0] * np.log10(x_fit) + coeffs[1])
    
    ax.plot(x_fit, y_fit, 'k--', linewidth=2, alpha=0.5, 
           label=f'Power law fit: t ∝ n^{coeffs[0]:.2f}')
    
    ax.set_xlabel('Circuit Size (constraints)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Proof Time (seconds)', fontweight='bold', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=11)
    
    # Add annotation
    ax.text(0.98, 0.02, f'Exponent: {coeffs[0]:.2f}\n(Sub-linear: <1.0)',
           transform=ax.transAxes, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
           fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scaling_curves.png', bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/scaling_curves.png")
    plt.close()


def plot_composite_comparison(df, output_dir='paper'):
    """Plot composite model comparison including CNN_ReLU as infeasible"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    comp_df = df[df['category'] == 'composite'].copy()
    
    # Average by model
    comp_stats = comp_df.groupby('model_base').agg({
        'proof_time_mean': 'mean',
        'circuit_size': 'first',
        'verify_time': 'mean'
    }).reset_index()
    
    # Add CNN_ReLU as failed
    failed_row = pd.DataFrame([{
        'model_base': 'CNN_ReLU',
        'proof_time_mean': np.nan,
        'circuit_size': np.nan,
        'verify_time': np.nan
    }])
    comp_stats = pd.concat([comp_stats, failed_row], ignore_index=True)
    comp_stats = comp_stats.sort_values('proof_time_mean')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Proof time comparison
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
    x_pos = range(len(comp_stats))
    
    bars = ax1.bar(x_pos, comp_stats['proof_time_mean'], color=colors,
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Mark failed
    bars[-1].set_hatch('///')
    bars[-1].set_alpha(0.3)
    
    # Add labels
    for i, (bar, time, model) in enumerate(zip(bars, comp_stats['proof_time_mean'], 
                                                 comp_stats['model_base'])):
        if pd.notna(time):
            ax1.text(bar.get_x() + bar.get_width()/2., time,
                    f'{time:.1f}s',
                    ha='center', va='bottom', fontweight='bold')
        else:
            ax1.text(bar.get_x() + bar.get_width()/2., 50,
                    'OOM\n(4 fails)',
                    ha='center', va='center', fontweight='bold', fontsize=10,
                    color='red')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(comp_stats['model_base'], rotation=0, ha='center', fontsize=14)
    ax1.set_ylabel('Proof Time (seconds)', fontweight='bold', fontsize=16)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, comp_stats['proof_time_mean'].max() * 1.2)
    
    # Circuit size comparison
    bars2 = ax2.bar(x_pos[:-1], comp_stats['circuit_size'][:-1] / 1e6, 
                    color=colors[:-1], alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, size in zip(bars2, comp_stats['circuit_size'][:-1]):
        ax2.text(bar.get_x() + bar.get_width()/2., size/1e6,
                f'{size/1e6:.2f}M',
                ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xticks(x_pos[:-1])
    ax2.set_xticklabels(comp_stats['model_base'][:-1], rotation=0, ha='center', fontsize=14)
    ax2.set_ylabel('Circuit Size (millions of constraints)', fontweight='bold', fontsize=16)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/composite_comparison.png', bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/composite_comparison.png")
    plt.close()


def plot_memory_analysis(df, output_dir='paper'):
    """Plot memory usage patterns"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Memory vs circuit size
    ax1.scatter(df['circuit_size'], df['peak_memory_mean_gb'], 
               s=100, alpha=0.6, c=df['proof_time_mean'], 
               cmap='viridis', edgecolors='black', linewidth=1)
    
    # Annotate outliers
    for _, row in df[df['peak_memory_mean_gb'] > 5].iterrows():
        ax1.annotate(row['model_base'], 
                    (row['circuit_size'], row['peak_memory_mean_gb']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('Circuit Size (constraints)', fontweight='bold', fontsize=16)
    ax1.set_ylabel('Peak Memory Usage (GB)', fontweight='bold', fontsize=16)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add 125GB line (system limit)
    ax1.axhline(y=125, color='r', linestyle='--', linewidth=2, alpha=0.7,
               label='System RAM limit (125GB)')
    ax1.legend()
    
    # Memory efficiency (constraints per GB)
    df['efficiency'] = df['circuit_size'] / df['peak_memory_mean_gb']
    
    efficiency_by_type = df.groupby(df['model_base'].apply(lambda x: 
        'Activation' if any(a in x for a in ['ReLU', 'SiLU', 'Tanh', 'Poly']) else
        'Pooling' if 'Pool' in x else
        'Normalization' if 'BatchNorm' in x else
        'Linear' if 'Dense' in x else
        'Composite'
    ))['efficiency'].mean().sort_values()
    
    colors_eff = ['#e74c3c' if x < 1e6 else '#f39c12' if x < 2e6 else '#2ecc71' 
                  for x in efficiency_by_type.values]
    
    bars = ax2.barh(efficiency_by_type.index, efficiency_by_type.values / 1e6,
                    color=colors_eff, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, efficiency_by_type.values):
        ax2.text(val/1e6, bar.get_y() + bar.get_height()/2.,
                f' {val/1e6:.2f}M',
                ha='left', va='center', fontweight='bold')
    
    ax2.set_xlabel('Memory Efficiency\n(million constraints per GB RAM)', fontweight='bold', fontsize=16)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/memory_analysis.png', bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/memory_analysis.png")
    plt.close()


def main():
    """Generate all visualizations"""
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    print()
    
    # Load data
    print("Loading results...")
    df = load_results()
    print(f"  Loaded {len(df)} experiments\n")
    
    # Create output directory (paper folder for LaTeX inclusion)
    output_dir = 'paper'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("Generating plots...")
    print()
    
    plot_tolerance_comparison(df, output_dir)
    plot_performance_tiers(df, output_dir)
    plot_layer_comparison(df, output_dir)
    plot_scaling_curves(df, output_dir)
    plot_composite_comparison(df, output_dir)
    plot_memory_analysis(df, output_dir)
    
    print()
    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll figures saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. tolerance_comparison.png - Precision impact (novel finding!)")
    print("  2. performance_tiers.png - Fast/Medium/Slow breakdown")
    print("  3. layer_comparison.png - Layer-by-layer ranking")
    print("  4. scaling_curves.png - Sub-linear scaling behavior")
    print("  5. composite_comparison.png - CNN architectures (with CNN_ReLU failure)")
    print("  6. memory_analysis.png - RAM usage patterns")
    print()
    print("Ready for paper submission!")
    print("="*80)


if __name__ == '__main__':
    main()
