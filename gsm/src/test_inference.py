"""
Test the inference pipeline components.
"""

import torch
import numpy as np
from refine_molecule import MoleculeRefiner


def test_molecule_refiner():
    """Test the molecule refiner with a simple example."""
    print("=" * 80)
    print("Testing MoleculeRefiner")
    print("=" * 80)
    
    # Test case 1: Ethanol (CCO)
    print("\nTest 1: Ethanol (CCO)")
    elements = ['C', 'C', 'O']
    
    # Approximate coordinates (with some noise)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.0, 1.4, 0.0],
    ]) + np.random.randn(3, 3) * 0.2
    
    refiner = MoleculeRefiner(verbose=True)
    mol, refined_coords, success = refiner.refine_to_molecule(
        coords, elements, guess_bonds=True
    )
    
    print(f"  Success: {success}")
    if mol is not None:
        from rdkit import Chem
        smiles = Chem.MolToSmiles(mol)
        print(f"  SMILES: {smiles}")
    
    # Test case 2: Benzene (C6H6 → 6 carbons after removing H)
    print("\nTest 2: Benzene ring (6 carbons)")
    elements = ['C'] * 6
    
    # Hexagon coordinates
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    coords = np.stack([
        1.4 * np.cos(angles),
        1.4 * np.sin(angles),
        np.zeros(6)
    ], axis=1)
    
    mol, refined_coords, success = refiner.refine_to_molecule(
        coords, elements, guess_bonds=True
    )
    
    print(f"  Success: {success}")
    if mol is not None:
        from rdkit import Chem
        smiles = Chem.MolToSmiles(mol)
        print(f"  SMILES: {smiles}")
    
    print("\n✓ MoleculeRefiner test complete!")


def test_euler_sampler():
    """Test the ODE sampler."""
    print("\n" + "=" * 80)
    print("Testing EulerSampler")
    print("=" * 80)
    
    from inference import EulerSampler
    
    sampler = EulerSampler(num_timesteps=10, t_start=0.01)
    
    print(f"  Number of timesteps: {sampler.num_timesteps}")
    print(f"  Timestep schedule: {sampler.steps[:5].tolist()} ... {sampler.steps[-3:].tolist()}")
    
    print("\n✓ EulerSampler test complete!")


def test_full_pipeline():
    """Test the full inference pipeline with a dummy model."""
    print("\n" + "=" * 80)
    print("Testing Full Inference Pipeline (with dummy model)")
    print("=" * 80)
    
    from model.model_factory import create_molecular_dit
    from inference import EulerSampler
    
    # Create a small model
    print("\nCreating model...")
    model = create_molecular_dit(
        hidden_size=128,
        num_heads=4,
        depth=2,
        max_atoms=20,
    )
    model.eval()
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy batch
    print("\nCreating dummy batch...")
    batch_size = 2
    max_atoms = 20
    
    batch = {
        'ref_pos': torch.randn(batch_size, max_atoms, 3),
        'ref_element': torch.randn(batch_size, max_atoms, 13),
        'atom_pad_mask': torch.ones(batch_size, max_atoms).bool(),
        'atom_resolved_mask': torch.ones(batch_size, max_atoms).bool(),
        'dreams_embedding': torch.randn(batch_size, 1024),
        'num_atoms': torch.tensor([15, 18]),
    }
    
    # Create sampler
    print("\nCreating sampler...")
    sampler = EulerSampler(num_timesteps=5, t_start=0.1)
    
    # Sample
    print("\nRunning sampling...")
    noise = torch.randn(batch_size, max_atoms, 3)
    
    with torch.no_grad():
        output = sampler.sample(
            model=model,
            noise=noise,
            batch=batch,
            show_progress=False,
        )
    
    coords = output['denoised_coords']
    print(f"  Output shape: {coords.shape}")
    print(f"  Output range: [{coords.min():.3f}, {coords.max():.3f}]")
    
    print("\n✓ Full pipeline test complete!")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("INFERENCE PIPELINE TESTS")
    print("=" * 80)
    
    try:
        test_molecule_refiner()
    except Exception as e:
        print(f"\n✗ MoleculeRefiner test failed: {e}")
    
    try:
        test_euler_sampler()
    except Exception as e:
        print(f"\n✗ EulerSampler test failed: {e}")
    
    try:
        test_full_pipeline()
    except Exception as e:
        print(f"\n✗ Full pipeline test failed: {e}")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
