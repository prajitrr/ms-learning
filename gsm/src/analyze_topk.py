"""
Top-K Analysis for Molecular Generation.

For each spectrum, we generate K candidates and evaluate:
- Top-K Tanimoto similarity (best match among K)
- Top-K connectivity match (ignoring stereochemistry)
- Top-K exact SMILES match
"""

import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator


def load_results(results_dir):
    """Load inference results."""
    results_dir = Path(results_dir)
    
    with open(results_dir / 'results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    return results


def compute_tanimoto_similarity(mol1, mol2):
    """Compute Morgan fingerprint Tanimoto similarity."""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp1 = mfpgen.GetFingerprint(mol1)
    fp2 = mfpgen.GetFingerprint(mol2)
    return Chem.DataStructs.TanimotoSimilarity(fp1, fp2)


def molecules_match_connectivity(mol1, mol2):
    """Check if two molecules have same connectivity (ignoring stereochemistry)."""
    # Remove stereochemistry
    mol1_copy = Chem.Mol(mol1)
    mol2_copy = Chem.Mol(mol2)
    Chem.RemoveStereochemistry(mol1_copy)
    Chem.RemoveStereochemistry(mol2_copy)
    
    # Compare canonical SMILES
    smiles1 = Chem.MolToSmiles(mol1_copy)
    smiles2 = Chem.MolToSmiles(mol2_copy)
    
    return smiles1 == smiles2


def analyze_topk(results_dir, k_values=[1, 5, 10]):
    """
    Analyze top-K performance.
    
    Args:
        results_dir: Directory containing inference results
        k_values: List of K values to evaluate
    """
    print("=" * 80)
    print("Top-K Analysis for Molecular Generation")
    print("=" * 80)
    
    results = load_results(results_dir)
    
    # Group by spectrum index
    samples_by_spectrum = defaultdict(list)
    for r in results:
        samples_by_spectrum[r['idx']].append(r)
    
    total_spectra = len(samples_by_spectrum)
    print(f"\nTotal spectra: {total_spectra}")
    print(f"Total structures generated: {len(results)}")
    
    # Filter to only successfully refined molecules
    valid_samples_by_spectrum = {}
    for idx, sample_list in samples_by_spectrum.items():
        valid = [s for s in sample_list if s['refinement_success']]
        if len(valid) > 0:
            valid_samples_by_spectrum[idx] = valid
    
    print(f"Spectra with ≥1 valid structure: {len(valid_samples_by_spectrum)}")
    
    # For each K value, compute metrics
    max_k = max(k_values)
    
    results_by_k = {}
    
    for k in k_values:
        print(f"\n{'='*80}")
        print(f"Top-{k} Evaluation")
        print(f"{'='*80}")
        
        # Metrics
        tanimoto_scores = []
        connectivity_matches = []
        exact_smiles_matches = []
        
        # For spectra that don't have enough valid samples
        insufficient_samples = 0
        
        for idx, sample_list in valid_samples_by_spectrum.items():
            # Get ground truth
            smiles_gt = sample_list[0]['smiles_gt']
            mol_gt = Chem.MolFromSmiles(smiles_gt)
            
            if mol_gt is None:
                continue
            
            # Check if we have at least K valid samples
            if len(sample_list) < k:
                insufficient_samples += 1
                # Use all available samples
                candidates = sample_list
            else:
                # Take top K (or all if less than K)
                candidates = sample_list[:k]
            
            # Compute metrics for each candidate
            best_tanimoto = 0.0
            has_connectivity_match = False
            has_exact_match = False
            
            for sample in candidates:
                mol_pred = sample['mol']
                
                # Tanimoto similarity
                tanimoto = compute_tanimoto_similarity(mol_pred, mol_gt)
                best_tanimoto = max(best_tanimoto, tanimoto)
                
                # Connectivity match (no stereochemistry)
                if molecules_match_connectivity(mol_pred, mol_gt):
                    has_connectivity_match = True
                
                # Exact SMILES match
                smiles_pred = Chem.MolToSmiles(mol_pred)
                smiles_gt_canonical = Chem.MolToSmiles(mol_gt)
                if smiles_pred == smiles_gt_canonical:
                    has_exact_match = True
            
            tanimoto_scores.append(best_tanimoto)
            connectivity_matches.append(1 if has_connectivity_match else 0)
            exact_smiles_matches.append(1 if has_exact_match else 0)
        
        # Compute statistics (over valid spectra only)
        n_evaluated = len(tanimoto_scores)
        
        print(f"\nEvaluated: {n_evaluated} spectra with valid molecules")
        print(f"  (Out of {total_spectra} total spectra)")
        if insufficient_samples > 0:
            print(f"  (Note: {insufficient_samples} spectra had <{k} valid samples)")
        
        print(f"\nTop-{k} Tanimoto Similarity (valid molecules only):")
        print(f"  Mean: {np.mean(tanimoto_scores):.4f}")
        print(f"  Median: {np.median(tanimoto_scores):.4f}")
        print(f"  Std: {np.std(tanimoto_scores):.4f}")
        print(f"  Min: {np.min(tanimoto_scores):.4f}")
        print(f"  Max: {np.max(tanimoto_scores):.4f}")
        
        connectivity_rate = np.mean(connectivity_matches)
        print(f"\nTop-{k} Connectivity Match Rate (valid only): {connectivity_rate:.2%}")
        print(f"  ({sum(connectivity_matches)}/{n_evaluated} spectra)")
        
        exact_match_rate = np.mean(exact_smiles_matches)
        print(f"\nTop-{k} Exact SMILES Match Rate (valid only): {exact_match_rate:.2%}")
        print(f"  ({sum(exact_smiles_matches)}/{n_evaluated} spectra)")
        
        # Compute metrics over ALL spectra (impute 0 for failed refinements)
        tanimoto_all = tanimoto_scores + [0.0] * (total_spectra - n_evaluated)
        connectivity_all = connectivity_matches + [0] * (total_spectra - n_evaluated)
        exact_all = exact_smiles_matches + [0] * (total_spectra - n_evaluated)
        
        print(f"\nTop-{k} Metrics (ALL spectra, failed=0):")
        print(f"  Mean Tanimoto: {np.mean(tanimoto_all):.4f}")
        print(f"  Connectivity Match Rate: {np.mean(connectivity_all):.2%}")
        print(f"    ({sum(connectivity_all)}/{total_spectra} spectra)")
        print(f"  Exact SMILES Match Rate: {np.mean(exact_all):.2%}")
        print(f"    ({sum(exact_all)}/{total_spectra} spectra)")
        
        # Store results
        results_by_k[k] = {
            'n_evaluated': n_evaluated,
            'total_spectra': total_spectra,
            'insufficient_samples': insufficient_samples,
            # Valid molecules only
            'mean_tanimoto_valid': float(np.mean(tanimoto_scores)),
            'median_tanimoto_valid': float(np.median(tanimoto_scores)),
            'std_tanimoto_valid': float(np.std(tanimoto_scores)),
            'connectivity_match_rate_valid': float(connectivity_rate),
            'exact_match_rate_valid': float(exact_match_rate),
            # All spectra (with imputation)
            'mean_tanimoto_all': float(np.mean(tanimoto_all)),
            'connectivity_match_rate_all': float(np.mean(connectivity_all)),
            'exact_match_rate_all': float(np.mean(exact_all)),
            'connectivity_matches': int(sum(connectivity_matches)),
            'exact_matches': int(sum(exact_smiles_matches)),
        }
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("Summary: Top-K Comparison (Valid Molecules Only)")
    print(f"{'='*80}")
    
    print(f"\n{'K':<5} {'Connectivity Match':<25} {'Mean Tanimoto':<20} {'Exact Match':<20}")
    print("-" * 75)
    for k in k_values:
        r = results_by_k[k]
        print(f"{k:<5} {r['connectivity_match_rate_valid']:.2%} ({r['connectivity_matches']}/{r['n_evaluated']}){'':>6} "
              f"{r['mean_tanimoto_valid']:.4f}{'':>15} {r['exact_match_rate_valid']:.2%} ({r['exact_matches']}/{r['n_evaluated']})")
    
    print(f"\n{'='*80}")
    print("Summary: Top-K Comparison (All Spectra, Failed=0)")
    print(f"{'='*80}")
    
    print(f"\n{'K':<5} {'Connectivity Match':<25} {'Mean Tanimoto':<20} {'Exact Match':<20}")
    print("-" * 75)
    for k in k_values:
        r = results_by_k[k]
        total = r['total_spectra']
        print(f"{k:<5} {r['connectivity_match_rate_all']:.2%} ({r['connectivity_matches']}/{total}){'':>6} "
              f"{r['mean_tanimoto_all']:.4f}{'':>15} {r['exact_match_rate_all']:.2%} ({r['exact_matches']}/{total})")
    
    # Save results
    output_path = Path(results_dir) / 'topk_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results_by_k, f, indent=2)
    
    print(f"\n✓ Top-K analysis saved to {output_path}")
    
    return results_by_k

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_topk.py <results_directory> [k1,k2,k3,...]")
        print("Example: python analyze_topk.py inference_results/val_predictions 1,5,10")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    # Parse K values
    if len(sys.argv) >= 3:
        k_values = [int(k.strip()) for k in sys.argv[2].split(',')]
    else:
        k_values = [1, 5, 10]
    
    analyze_topk(results_dir, k_values=k_values)
    
    print("\n" + "=" * 80)
    print("✓ Top-K analysis complete!")
    print("=" * 80)
