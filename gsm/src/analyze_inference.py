"""
Analyze inference results and compute metrics.

Compares generated structures against ground truth.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def load_results(results_dir):
    """Load inference results."""
    results_dir = Path(results_dir)
    
    with open(results_dir / 'results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    return results


def compute_rmsd(coords1, coords2):
    """
    Compute RMSD between two coordinate sets.
    
    Args:
        coords1, coords2: [N, 3] numpy arrays
        
    Returns:
        RMSD value
    """
    # Center both
    coords1 = coords1 - coords1.mean(axis=0)
    coords2 = coords2 - coords2.mean(axis=0)
    
    # Compute RMSD
    rmsd = np.sqrt(((coords1 - coords2) ** 2).sum() / len(coords1))
    return rmsd


def smiles_to_coords(smiles, use_mmff=True):
    """Convert SMILES to 3D coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        return None, None
    
    if use_mmff:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    
    mol = Chem.RemoveHs(mol)
    
    # Get coords
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    
    return mol, coords


def compute_molecular_metrics(mol_pred, mol_gt):
    """
    Compute molecular similarity metrics.
    
    Args:
        mol_pred: Predicted RDKit mol
        mol_gt: Ground truth RDKit mol
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Molecular formula
    formula_pred = Chem.rdMolDescriptors.CalcMolFormula(mol_pred)
    formula_gt = Chem.rdMolDescriptors.CalcMolFormula(mol_gt)
    metrics['formula_match'] = (formula_pred == formula_gt)
    
    # Molecular weight
    mw_pred = Descriptors.MolWt(mol_pred)
    mw_gt = Descriptors.MolWt(mol_gt)
    metrics['mw_error'] = abs(mw_pred - mw_gt)
    
    # Number of atoms
    metrics['n_atoms_pred'] = mol_pred.GetNumAtoms()
    metrics['n_atoms_gt'] = mol_gt.GetNumAtoms()
    metrics['n_atoms_match'] = (metrics['n_atoms_pred'] == metrics['n_atoms_gt'])
    
    # Fingerprint similarity (Morgan) - using new API
    from rdkit.Chem import rdFingerprintGenerator
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_pred = mfpgen.GetFingerprint(mol_pred)
    fp_gt = mfpgen.GetFingerprint(mol_gt)
    metrics['morgan_similarity'] = Chem.DataStructs.TanimotoSimilarity(fp_pred, fp_gt)
    
    # Connectivity match (ignoring stereochemistry)
    # Morgan similarity of 1.0 means identical connectivity
    metrics['connectivity_match'] = (metrics['morgan_similarity'] >= 0.9999)
    
    # SMILES comparison (multiple methods)
    smiles_pred = Chem.MolToSmiles(mol_pred)
    smiles_gt = Chem.MolToSmiles(mol_gt)
    metrics['smiles_match'] = (smiles_pred == smiles_gt)
    
    # Remove stereochemistry and compare canonical SMILES
    # Make copies first since RemoveStereochemistry modifies in place
    mol_pred_no_stereo = Chem.Mol(mol_pred)
    mol_gt_no_stereo = Chem.Mol(mol_gt)
    Chem.RemoveStereochemistry(mol_pred_no_stereo)
    Chem.RemoveStereochemistry(mol_gt_no_stereo)
    
    smiles_pred_no_stereo = Chem.MolToSmiles(mol_pred_no_stereo)
    smiles_gt_no_stereo = Chem.MolToSmiles(mol_gt_no_stereo)
    metrics['smiles_match_no_stereo'] = (smiles_pred_no_stereo == smiles_gt_no_stereo)
    
    # InChI comparison (structure-based, more robust)
    try:
        inchi_pred = Chem.MolToInchi(mol_pred)
        inchi_gt = Chem.MolToInchi(mol_gt)
        metrics['inchi_match'] = (inchi_pred == inchi_gt)
        
        # InChI without stereochemistry (connectivity only)
        inchikey_pred = Chem.MolToInchiKey(mol_pred)
        inchikey_gt = Chem.MolToInchiKey(mol_gt)
        # First block of InChI Key is connectivity only
        metrics['inchikey_connectivity_match'] = (inchikey_pred.split('-')[0] == inchikey_gt.split('-')[0])
    except:
        metrics['inchi_match'] = False
        metrics['inchikey_connectivity_match'] = False
    
    return metrics


def analyze_results(results_dir):
    """
    Analyze inference results.
    """
    print("=" * 80)
    print("Analyzing inference results...")
    print("=" * 80)
    
    results = load_results(results_dir)
    
    print(f"\nTotal structures: {len(results)}")
    
    # Refinement statistics
    num_refined = sum(r['refinement_success'] for r in results)
    print(f"Successfully refined: {num_refined} ({100*num_refined/len(results):.1f}%)")
    
    # Group by original sample
    samples = defaultdict(list)
    for r in results:
        samples[r['idx']].append(r)
    
    print(f"Unique spectra: {len(samples)}")
    print(f"Samples per spectrum: {len(results) / len(samples):.1f}")
    
    # Compute metrics for successfully refined molecules
    print("\n" + "=" * 80)
    print("Computing molecular metrics...")
    print("=" * 80)
    
    all_metrics = []
    rmsd_values = []
    
    for idx, sample_results in samples.items():
        # Get ground truth
        smiles_gt = sample_results[0]['smiles_gt']
        mol_gt, coords_gt = smiles_to_coords(smiles_gt)
        
        if mol_gt is None:
            print(f"  Warning: Could not generate GT structure for {smiles_gt}")
            continue
        
        for r in sample_results:
            if not r['refinement_success']:
                continue
            
            mol_pred = r['mol']
            coords_pred = r['coords_refined']
            
            # Compute metrics
            metrics = compute_molecular_metrics(mol_pred, mol_gt)
            metrics['idx'] = idx
            metrics['sample_idx'] = r['sample_idx']
            
            # Compute RMSD if same number of atoms
            if len(coords_pred) == len(coords_gt):
                rmsd = compute_rmsd(coords_pred, coords_gt)
                metrics['rmsd'] = rmsd
                rmsd_values.append(rmsd)
            else:
                metrics['rmsd'] = None
            
            all_metrics.append(metrics)
    
    # Aggregate statistics
    print(f"\nMetrics computed for {len(all_metrics)} successfully refined structures")
    
    if len(all_metrics) > 0:
        # Formula match rate
        formula_match_rate = sum(m['formula_match'] for m in all_metrics) / len(all_metrics)
        print(f"\nExact Matches:")
        print(f"  Formula match rate: {formula_match_rate:.2%}")
        
        # SMILES match rate (exact structure match including stereochemistry)
        smiles_match_rate = sum(m['smiles_match'] for m in all_metrics) / len(all_metrics)
        print(f"  SMILES match rate (with stereo): {smiles_match_rate:.2%}")
        
        # SMILES match without stereochemistry
        smiles_match_no_stereo_rate = sum(m['smiles_match_no_stereo'] for m in all_metrics) / len(all_metrics)
        print(f"  SMILES match rate (no stereo): {smiles_match_no_stereo_rate:.2%}")
        
        # Connectivity match (Morgan similarity >= 0.9999)
        connectivity_match_rate = sum(m['connectivity_match'] for m in all_metrics) / len(all_metrics)
        print(f"  Connectivity match rate (Morgan=1.0): {connectivity_match_rate:.2%}")
        
        # InChI Key connectivity match
        inchikey_connectivity_rate = sum(m['inchikey_connectivity_match'] for m in all_metrics) / len(all_metrics)
        print(f"  InChI Key connectivity match: {inchikey_connectivity_rate:.2%}")
        
        # Atom count accuracy
        atom_match_rate = sum(m['n_atoms_match'] for m in all_metrics) / len(all_metrics)
        print(f"  Atom count match rate: {atom_match_rate:.2%}")
        
        # Morgan similarity
        morgan_sims = [m['morgan_similarity'] for m in all_metrics]
        print(f"\nMorgan similarity (Tanimoto):")
        print(f"  Mean: {np.mean(morgan_sims):.3f}")
        print(f"  Median: {np.median(morgan_sims):.3f}")
        print(f"  Std: {np.std(morgan_sims):.3f}")
        
        # RMSD (for structures with matching atom count)
        if len(rmsd_values) > 0:
            print(f"\nRMSD (matching atom count only, n={len(rmsd_values)}):")
            print(f"  Mean: {np.mean(rmsd_values):.3f} Å")
            print(f"  Median: {np.median(rmsd_values):.3f} Å")
            print(f"  Std: {np.std(rmsd_values):.3f} Å")
            print(f"  Min: {np.min(rmsd_values):.3f} Å")
            print(f"  Max: {np.max(rmsd_values):.3f} Å")
        
        # Molecular weight error
        mw_errors = [m['mw_error'] for m in all_metrics]
        print(f"\nMolecular weight error:")
        print(f"  Mean: {np.mean(mw_errors):.3f} Da")
        print(f"  Median: {np.median(mw_errors):.3f} Da")
        print(f"  Max: {np.max(mw_errors):.3f} Da")
    
    # Save metrics
    metrics_path = Path(results_dir) / 'metrics.json'
    with open(metrics_path, 'w') as f:
        # Convert to JSON-serializable format
        metrics_json = []
        for m in all_metrics:
            m_json = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                     for k, v in m.items() if v is not None}
            metrics_json.append(m_json)
        json.dump(metrics_json, f, indent=2)
    
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    # Summary statistics
    summary = {
        'total_structures': len(results),
        'successfully_refined': num_refined,
        'refinement_rate': num_refined / len(results),
        'unique_spectra': len(samples),
        'samples_per_spectrum': len(results) / len(samples),
    }
    
    if len(all_metrics) > 0:
        summary.update({
            'formula_match_rate': formula_match_rate,
            'smiles_match_rate': smiles_match_rate,
            'smiles_match_no_stereo_rate': smiles_match_no_stereo_rate,
            'connectivity_match_rate': connectivity_match_rate,
            'inchikey_connectivity_rate': inchikey_connectivity_rate,
            'atom_match_rate': atom_match_rate,
            'mean_morgan_similarity': float(np.mean(morgan_sims)),
            'median_morgan_similarity': float(np.median(morgan_sims)),
            'mean_mw_error': float(np.mean(mw_errors)),
        })
        
        if len(rmsd_values) > 0:
            summary.update({
                'mean_rmsd': float(np.mean(rmsd_values)),
                'median_rmsd': float(np.median(rmsd_values)),
                'n_rmsd_computed': len(rmsd_values),
            })
    
    summary_path = Path(results_dir) / 'summary_metrics.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to {summary_path}")
    
    return all_metrics, summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_inference.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    analyze_results(results_dir)
    
    print("\n" + "=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80)
