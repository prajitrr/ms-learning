"""
Example usage of MolecularDataProcessor for processing SMILES strings.
"""

import torch
from molecular_data_processor import MolecularDataProcessor


def example_basic_usage():
    """Basic example of processing SMILES strings."""
    
    # Initialize processor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = MolecularDataProcessor(
        device=device,
        scale=16.0,
        ref_scale=5.0,
        multiplicity=1,
        inference_multiplicity=1,
        apply_rotation=True,
        apply_conformer_sampling=True,
        backend="torch"
    )
    
    # Example SMILES strings
    smiles_list = [
        'CCO',  # Ethanol
        'CC(=O)O',  # Acetic acid
        'c1ccccc1',  # Benzene
    ]
    
    # Process batch for training
    batch = {'smiles': smiles_list}
    batch = processor.preprocess_training(batch)
    
    print("Training batch keys:", batch.keys())
    print("Coords shape:", batch['coords'].shape)
    print("Elements shape:", batch['elements'].shape)
    print("Mask shape:", batch['atom_pad_mask'].shape)
    
    return batch


def example_with_multiplicity():
    """Example with batch multiplicity for data augmentation."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = MolecularDataProcessor(
        device=device,
        scale=16.0,
        multiplicity=4,  # Each sample repeated 4 times with different augmentations
        apply_rotation=True,
        apply_conformer_sampling=True,
    )
    
    smiles_list = ['CCO', 'CC(=O)O']
    batch = {'smiles': smiles_list}
    batch = processor.preprocess_training(batch)
    
    print(f"\nOriginal batch size: 2")
    print(f"Effective batch size with multiplicity=4: {batch['coords'].shape[0]}")
    
    return batch


def example_conformer_sampling():
    """Example demonstrating conformer sampling."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = MolecularDataProcessor(
        device=device,
        apply_conformer_sampling=True,
    )
    
    # Generate multiple conformers for the same molecule
    smiles = 'CCCCCCCC'  # Octane - flexible molecule
    
    coords_list = []
    for i in range(5):
        coords, elements, mol = processor.sample_low_energy_conformer(smiles)
        coords_list.append(coords)
        print(f"Conformer {i+1} shape: {coords.shape}")
    
    # Check that different conformers were generated
    import numpy as np
    for i in range(1, len(coords_list)):
        diff = np.abs(coords_list[i] - coords_list[0]).max()
        print(f"Max difference from conformer 1 to {i+1}: {diff:.4f} Å")
    
    return coords_list


def example_inference():
    """Example of processing for inference (no augmentation)."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = MolecularDataProcessor(
        device=device,
        apply_rotation=False,  # No rotation during inference
        apply_conformer_sampling=False,  # Deterministic for inference
    )
    
    smiles_list = ['CCO', 'CC(=O)O']
    batch = {'smiles': smiles_list}
    batch = processor.preprocess_inference(batch)
    
    print("\nInference batch keys:", batch.keys())
    print("Coords shape:", batch['coords'].shape)
    
    return batch


def example_with_precomputed_coords():
    """Example using pre-computed coordinates instead of SMILES."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = MolecularDataProcessor(device=device)
    
    # Pre-computed coordinates (e.g., from a dataset)
    coords = torch.randn(2, 10, 3)  # 2 molecules, max 10 atoms, 3D coords
    atom_pad_mask = torch.ones(2, 10, dtype=torch.bool)
    atom_pad_mask[0, 8:] = False  # First molecule has 8 atoms
    atom_pad_mask[1, 6:] = False  # Second molecule has 6 atoms
    
    batch = {
        'coords': coords,
        'atom_pad_mask': atom_pad_mask,
    }
    
    batch = processor.preprocess_training(batch)
    
    print("\nPre-computed coords batch shape:", batch['coords'].shape)
    
    return batch


if __name__ == '__main__':
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    example_basic_usage()
    
    print("\n" + "=" * 60)
    print("Example 2: Batch Multiplicity")
    print("=" * 60)
    example_with_multiplicity()
    
    print("\n" + "=" * 60)
    print("Example 3: Conformer Sampling")
    print("=" * 60)
    example_conformer_sampling()
    
    print("\n" + "=" * 60)
    print("Example 4: Inference Mode")
    print("=" * 60)
    example_inference()
    
    print("\n" + "=" * 60)
    print("Example 5: Pre-computed Coordinates")
    print("=" * 60)
    example_with_precomputed_coords()
