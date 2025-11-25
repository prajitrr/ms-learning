"""
Universal data loader for exported MassSpecGym data.

This loader works in any environment (no DreaMS dependency needed).
Use this in the SimpleFold/Lightning environment.
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ExportedMassSpecGymDataset(torch.utils.data.Dataset):
    """
    Simple dataset loader for exported MassSpecGym data.
    
    No DreaMS dependency required - works with standard h5py.
    """
    
    def __init__(
        self,
        h5_path: str,
        preload_to_memory: bool = True,
    ):
        """
        Initialize dataset from exported HDF5 file.
        
        Args:
            h5_path: Path to exported .h5 file (train.h5 or val.h5)
            preload_to_memory: Whether to load all data to memory (faster but uses RAM)
        """
        super().__init__()
        
        self.h5_path = Path(h5_path)
        self.preload_to_memory = preload_to_memory
        
        if not self.h5_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {h5_path}")
        
        # Load metadata
        with h5py.File(self.h5_path, 'r') as f:
            self.n_samples = f.attrs['n_samples']
            self.split = f.attrs['split']
            self.embedding_dim = f.attrs['embedding_dim']
            self.has_cached_structures = f.attrs.get('has_cached_structures', False)
            if self.has_cached_structures:
                self.max_atoms = f.attrs['max_atoms']
        
        print(f"Loaded {self.split} dataset: {self.n_samples} samples")
        if self.has_cached_structures:
            print(f"  ✓ Has cached 3D structures (max_atoms={self.max_atoms})")
        
        # Optionally preload to memory
        if preload_to_memory:
            print(f"Preloading data to memory...")
            with h5py.File(self.h5_path, 'r') as f:
                # Decode bytes to strings
                self.smiles = [
                    s.decode('utf-8') if isinstance(s, bytes) else s 
                    for s in f['smiles'][:]
                ]
                self.dreams_embeddings = torch.from_numpy(
                    f['dreams_embeddings'][:].astype(np.float32)
                )
                self.original_indices = f['original_indices'][:]
                
                # Load cached structures if available
                if self.has_cached_structures:
                    self.coords = torch.from_numpy(f['coords'][:].astype(np.float32))
                    self.elements = f['elements'][:]  # String array
                    self.n_atoms = torch.from_numpy(f['n_atoms'][:].astype(np.int64))
                else:
                    self.coords = None
                    self.elements = None
                    self.n_atoms = None
            print(f"✓ Data preloaded")
        else:
            self.smiles = None
            self.dreams_embeddings = None
            self.original_indices = None
            self.coords = None
            self.elements = None
            self.n_atoms = None
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a single sample.
        
        Returns:
            dict with keys:
                - smiles: SMILES string
                - dreams_embedding: [1024] tensor
                - original_idx: int
                - idx: int (index in this split)
                - coords: [max_atoms, 3] tensor (if cached)
                - elements: [max_atoms] array of element symbols (if cached)
                - n_atoms: int (if cached)
        """
        if self.preload_to_memory:
            # Load from memory
            smiles = self.smiles[idx]
            dreams_embedding = self.dreams_embeddings[idx]
            original_idx = self.original_indices[idx]
            
            result = {
                'smiles': smiles,
                'dreams_embedding': dreams_embedding,
                'original_idx': int(original_idx),
                'idx': idx,
            }
            
            # Add cached structures if available
            if self.has_cached_structures:
                result['coords'] = self.coords[idx]
                result['elements'] = self.elements[idx]
                result['n_atoms'] = self.n_atoms[idx]
            
            return result
        else:
            # Load from disk
            with h5py.File(self.h5_path, 'r') as f:
                smiles = f['smiles'][idx]
                # Decode bytes to string
                if isinstance(smiles, bytes):
                    smiles = smiles.decode('utf-8')
                dreams_embedding = torch.from_numpy(
                    f['dreams_embeddings'][idx].astype(np.float32)
                )
                original_idx = f['original_indices'][idx]
                
                result = {
                    'smiles': smiles,
                    'dreams_embedding': dreams_embedding,
                    'original_idx': int(original_idx),
                    'idx': idx,
                }
                
                # Add cached structures if available
                if self.has_cached_structures:
                    result['coords'] = torch.from_numpy(
                        f['coords'][idx].astype(np.float32)
                    )
                    result['elements'] = f['elements'][idx]
                    result['n_atoms'] = int(f['n_atoms'][idx])
                
                return result


def load_exported_data(
    data_dir: str,
    split: str = 'train',
    preload: bool = True,
) -> ExportedMassSpecGymDataset:
    """
    Convenience function to load exported data.
    
    Args:
        data_dir: Directory containing train.h5 and val.h5
        split: 'train' or 'val'
        preload: Whether to preload to memory
        
    Returns:
        ExportedMassSpecGymDataset
    """
    h5_path = Path(data_dir) / f"{split}.h5"
    return ExportedMassSpecGymDataset(h5_path, preload_to_memory=preload)


if __name__ == "__main__":
    # Test the loader
    import sys
    
    # Example usage
    data_dir = "../data/massspecgym_exported"
    
    print("Testing data loader...")
    print("="*80)
    
    # Load train split
    train_dataset = load_exported_data(data_dir, split='train', preload=True)
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    
    # Load a sample
    sample = train_dataset[0]
    print(f"\nSample 0:")
    print(f"  SMILES: {sample['smiles']}")
    print(f"  DreaMS embedding shape: {sample['dreams_embedding'].shape}")
    print(f"  DreaMS embedding dtype: {sample['dreams_embedding'].dtype}")
    print(f"  Original index: {sample['original_idx']}")
    
    # Load val split
    val_dataset = load_exported_data(data_dir, split='val', preload=True)
    print(f"\nVal dataset size: {len(val_dataset)}")
    
    # Test batch loading
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        return {
            'smiles': [b['smiles'] for b in batch],
            'dreams_embedding': torch.stack([b['dreams_embedding'] for b in batch]),
            'original_idx': torch.tensor([b['original_idx'] for b in batch]),
        }
    
    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    batch = next(iter(loader))
    print(f"\nBatch test:")
    print(f"  Batch size: {len(batch['smiles'])}")
    print(f"  DreaMS embeddings shape: {batch['dreams_embedding'].shape}")
    
    print("\n✓ All tests passed!")
