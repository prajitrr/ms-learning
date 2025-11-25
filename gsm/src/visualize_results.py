"""
Visualize inference results.

Creates plots and molecular visualizations.
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_metrics(results_dir, output_dir=None):
    """
    Create plots of inference metrics.
    
    Args:
        results_dir: Directory containing inference results
        output_dir: Output directory for plots (defaults to results_dir/plots)
    """
    results_dir = Path(results_dir)
    
    if output_dir is None:
        output_dir = results_dir / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    with open(results_dir / 'metrics.json', 'r') as f:
        metrics = json.load(f)
    
    if len(metrics) == 0:
        print("No metrics to plot")
        return
    
    # Extract values
    morgan_sims = [m['morgan_similarity'] for m in metrics]
    mw_errors = [m['mw_error'] for m in metrics]
    rmsd_values = [m['rmsd'] for m in metrics if 'rmsd' in m]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Molecular Structure Generation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Morgan Similarity Distribution
    ax = axes[0, 0]
    ax.hist(morgan_sims, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(morgan_sims), color='r', linestyle='--', 
               label=f'Mean: {np.mean(morgan_sims):.3f}')
    ax.axvline(np.median(morgan_sims), color='g', linestyle='--',
               label=f'Median: {np.median(morgan_sims):.3f}')
    ax.set_xlabel('Morgan Fingerprint Similarity (Tanimoto)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Molecular Similarity Distribution', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Molecular Weight Error
    ax = axes[0, 1]
    ax.hist(mw_errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(np.mean(mw_errors), color='r', linestyle='--',
               label=f'Mean: {np.mean(mw_errors):.2f} Da')
    ax.set_xlabel('Molecular Weight Error (Da)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Molecular Weight Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: RMSD Distribution (if available)
    ax = axes[1, 0]
    if len(rmsd_values) > 0:
        ax.hist(rmsd_values, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(np.mean(rmsd_values), color='r', linestyle='--',
                   label=f'Mean: {np.mean(rmsd_values):.3f} Å')
        ax.set_xlabel('RMSD (Å)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Geometric Accuracy (n={len(rmsd_values)})', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No RMSD data\n(requires matching atom counts)', 
                ha='center', va='center', fontsize=12)
        ax.set_title('RMSD Distribution', fontweight='bold')
    
    # Plot 4: Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate statistics
    formula_match = sum(m['formula_match'] for m in metrics) / len(metrics)
    smiles_match = sum(m['smiles_match'] for m in metrics) / len(metrics)
    atom_match = sum(m['n_atoms_match'] for m in metrics) / len(metrics)
    
    stats_text = f"""
    Summary Statistics
    {'='*40}
    
    Total Structures:  {len(metrics)}
    
    Exact Matches:
      Formula:         {formula_match:.1%}
      SMILES:          {smiles_match:.1%}
      Atom Count:      {atom_match:.1%}
    
    Morgan Similarity:
      Mean:            {np.mean(morgan_sims):.3f}
      Median:          {np.median(morgan_sims):.3f}
      Std:             {np.std(morgan_sims):.3f}
    
    Molecular Weight Error:
      Mean:            {np.mean(mw_errors):.2f} Da
      Median:          {np.median(mw_errors):.2f} Da
    """
    
    if len(rmsd_values) > 0:
        stats_text += f"""
    RMSD (matching atoms):
      Mean:            {np.mean(rmsd_values):.3f} Å
      Median:          {np.median(rmsd_values):.3f} Å
      n:               {len(rmsd_values)}
        """
    
    ax.text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=10,
            verticalalignment='center')
    
    plt.tight_layout()
    
    # Save figure
    plot_path = output_dir / 'metrics_summary.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {plot_path}")
    
    plt.close()
    
    # Additional plots
    plot_correlation_heatmap(metrics, output_dir)
    plot_per_sample_metrics(metrics, output_dir)


def plot_correlation_heatmap(metrics, output_dir):
    """Plot correlation between different metrics."""
    import pandas as pd
    import seaborn as sns
    
    # Create dataframe
    df = pd.DataFrame([
        {
            'morgan_sim': m['morgan_similarity'],
            'mw_error': m['mw_error'],
            'rmsd': m.get('rmsd', np.nan),
            'formula_match': int(m['formula_match']),
            'smiles_match': int(m['smiles_match']),
        }
        for m in metrics
    ])
    
    # Compute correlation
    corr = df.corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Metric Correlations', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plot_path = output_dir / 'correlation_heatmap.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Correlation heatmap saved to {plot_path}")
    plt.close()


def plot_per_sample_metrics(metrics, output_dir, max_samples=50):
    """Plot metrics for individual samples."""
    # Group by sample idx
    from collections import defaultdict
    samples = defaultdict(list)
    for m in metrics:
        samples[m['idx']].append(m)
    
    # Get first N samples
    sample_indices = sorted(samples.keys())[:max_samples]
    
    # Extract Morgan similarity for each sample
    morgan_means = []
    morgan_stds = []
    
    for idx in sample_indices:
        sims = [m['morgan_similarity'] for m in samples[idx]]
        morgan_means.append(np.mean(sims))
        morgan_stds.append(np.std(sims) if len(sims) > 1 else 0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = np.arange(len(sample_indices))
    ax.errorbar(x, morgan_means, yerr=morgan_stds, fmt='o-', capsize=3,
                alpha=0.7, label='Morgan Similarity')
    ax.axhline(np.mean(morgan_means), color='r', linestyle='--',
               label=f'Overall Mean: {np.mean(morgan_means):.3f}')
    
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Morgan Similarity', fontsize=11)
    ax.set_title(f'Per-Sample Morgan Similarity (first {len(sample_indices)} samples)',
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plot_path = output_dir / 'per_sample_metrics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-sample plot saved to {plot_path}")
    plt.close()


def visualize_molecule_3d(mol, coords, output_path=None):
    """
    Create 3D visualization of molecule.
    
    Requires py3Dmol for interactive visualization.
    """
    try:
        import py3Dmol
        from rdkit import Chem
        
        # Create viewer
        view = py3Dmol.view(width=400, height=400)
        
        # Add molecule
        mol_block = Chem.MolToMolBlock(mol)
        view.addModel(mol_block, 'sdf')
        
        # Style
        view.setStyle({'stick': {}})
        view.setBackgroundColor('white')
        view.zoomTo()
        
        # Save or display
        if output_path:
            html = view._make_html()
            with open(output_path, 'w') as f:
                f.write(html)
            print(f"✓ 3D visualization saved to {output_path}")
        else:
            return view
            
    except ImportError:
        print("py3Dmol not installed. Install with: pip install py3Dmol")
        return None


def create_visualization_html(results_dir, max_molecules=20):
    """
    Create HTML page with embedded molecule visualizations.
    """
    results_dir = Path(results_dir)
    
    # Load results
    with open(results_dir / 'results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    # Filter successful refinements
    successful = [r for r in results if r['refinement_success']][:max_molecules]
    
    if len(successful) == 0:
        print("No successfully refined molecules to visualize")
        return
    
    print(f"Creating HTML visualization for {len(successful)} molecules...")
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Molecular Structure Generation Results</title>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .molecule { 
                display: inline-block; 
                margin: 10px; 
                border: 1px solid #ccc; 
                padding: 10px;
                border-radius: 5px;
            }
            .viewer { width: 300px; height: 300px; }
            .info { 
                margin-top: 10px; 
                font-size: 12px;
                max-width: 300px;
            }
            h1 { color: #333; }
            .stats { background: #f0f0f0; padding: 10px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Generated Molecular Structures</h1>
        <div class="stats">
            <p><strong>Total molecules:</strong> """ + str(len(successful)) + """</p>
        </div>
        <div id="molecules">
    """
    
    from rdkit import Chem
    
    for i, r in enumerate(successful):
        mol_block = Chem.MolToMolBlock(r['mol'])
        mol_block_escaped = mol_block.replace('\n', '\\n')
        
        html += f"""
        <div class="molecule">
            <div id="viewer{i}" class="viewer"></div>
            <div class="info">
                <strong>Sample {r['idx']}</strong><br>
                GT: {r['smiles_gt']}<br>
                Atoms: {r['n_atoms']}<br>
            </div>
        </div>
        <script>
            var viewer{i} = $3Dmol.createViewer("viewer{i}");
            viewer{i}.addModel("{mol_block_escaped}", "sdf");
            viewer{i}.setStyle({{}}, {{stick: {{}}}});
            viewer{i}.setBackgroundColor('white');
            viewer{i}.zoomTo();
            viewer{i}.render();
        </script>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML
    html_path = results_dir / 'visualization.html'
    with open(html_path, 'w') as f:
        f.write(html)
    
    print(f"✓ HTML visualization saved to {html_path}")
    print(f"  Open in browser to view 3D structures")


def main():
    """Main visualization function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    print("=" * 80)
    print("Creating visualizations...")
    print("=" * 80)
    
    # Create plots
    plot_metrics(results_dir)
    
    # Create HTML visualization
    try:
        create_visualization_html(results_dir)
    except Exception as e:
        print(f"Could not create HTML visualization: {e}")
    
    print("\n" + "=" * 80)
    print("✓ Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
