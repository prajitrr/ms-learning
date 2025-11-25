#
# Mass Spectrum to Molecular Structure Training DataModule
# Adapted from SimpleFold for GSM (Generative Spectrum-to-Molecule)
#

import os
import json
import pickle
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
from torch import Tensor
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from rdkit import Chem
from rdkit.Chem import AllChem

# No DreaMS dependency - we load from exported HDF5 files
try:
    from datasets.exported_data_loader import ExportedMassSpecGymDataset
except ImportError:
    from exported_data_loader import ExportedMassSpecGymDataset


# Unique atoms in MassSpecGym dataset
ATOM_TYPES = ['B', 'Se', 'O', 'F', 'C', 'Cl', 'Br', 'Si', 'P', 'N', 'As', 'S', 'I']
ATOM_TO_IDX = {atom: idx for idx, atom in enumerate(ATOM_TYPES)}
IDX_TO_ATOM = {idx: atom for atom, idx in ATOM_TO_IDX.items()}
NUM_ATOM_TYPES = len(ATOM_TYPES)

# Dataset configuration
MAX_ATOMS = 80  # Maximum number of atoms (SMILES length 72, add buffer)
DREAMS_EMBEDDING_DIM = 1024

# IMPORTANT: Molecules are PERMUTATION-INVARIANT
# We do NOT use positional embeddings based on atom order
# Only spatial relationships matter (encoded via 3D coordinates + element type)


def smiles_to_3d_coords(
    smiles: str,
    use_mmff: bool = True,
    max_iters: int = 200,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Convert SMILES string to 3D coordinates using RDKit.
    
    NOTE: Generates 3D structure WITH hydrogens for accurate optimization,
    then returns only HEAVY ATOMS (no hydrogens).
    
    Args:
        smiles: SMILES string
        use_mmff: Whether to use MMFF force field (True) or UFF (False)
        max_iters: Maximum iterations for optimization
        random_seed: Random seed for reproducibility
        
    Returns:
        coords: (N, 3) numpy array of atom coordinates (heavy atoms only)
        elements: (N,) numpy array of atomic element indices (heavy atoms only)
        success: Whether conversion was successful
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, False
        
        # Add hydrogens for accurate geometry optimization
        mol_with_h = Chem.AddHs(mol)
        
        # Generate 3D coordinates (with hydrogens)
        if random_seed is not None:
            result = AllChem.EmbedMolecule(mol_with_h, randomSeed=random_seed)
        else:
            result = AllChem.EmbedMolecule(mol_with_h, randomSeed=-1)
        
        if result != 0:
            # Embedding failed, try without random seed
            result = AllChem.EmbedMolecule(mol_with_h, randomSeed=-1)
            if result != 0:
                return None, None, False
        
        # Optimize geometry (with hydrogens for accurate bond angles)
        try:
            if use_mmff:
                AllChem.MMFFOptimizeMolecule(mol_with_h, maxIters=max_iters)
            else:
                AllChem.UFFOptimizeMolecule(mol_with_h, maxIters=max_iters)
        except:
            # If optimization fails, continue with unoptimized structure
            pass
        
        # Remove hydrogens - keep only heavy atoms with optimized positions
        mol_heavy = Chem.RemoveHs(mol_with_h)
        
        # Extract coordinates (heavy atoms only)
        conf = mol_heavy.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol_heavy.GetNumAtoms())])
        
        # Extract atomic element indices (heavy atoms only)
        elements = []
        for atom in mol_heavy.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol == 'H':
                # Skip any remaining hydrogens (safety check)
                continue
            if symbol in ATOM_TO_IDX:
                elements.append(ATOM_TO_IDX[symbol])
            else:
                # Unknown atom type, skip this molecule
                return None, None, False
        elements = np.array(elements, dtype=np.int64)
        
        # Verify we removed all hydrogens
        if len(coords) != len(elements):
            print(f"Warning: Coordinate/element mismatch for {smiles}")
            return None, None, False
        
        return coords, elements, True
        
    except Exception as e:
        print(f"Error converting SMILES {smiles}: {e}")
        return None, None, False


def pad_atom_features(
    coords: np.ndarray,
    elements: np.ndarray,
    max_atoms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad atom features to max_atoms.
    
    Args:
        coords: (N, 3) atom coordinates
        elements: (N,) atom element indices
        max_atoms: Maximum number of atoms
        
    Returns:
        padded_coords: (max_atoms, 3)
        padded_elements: (max_atoms,)
        atom_mask: (max_atoms,) boolean mask of valid atoms
    """
    n_atoms = len(coords)
    
    if n_atoms > max_atoms:
        # Truncate if too many atoms
        coords = coords[:max_atoms]
        elements = elements[:max_atoms]
        n_atoms = max_atoms
    
    # Create padded arrays
    padded_coords = np.zeros((max_atoms, 3), dtype=np.float32)
    padded_elements = np.zeros(max_atoms, dtype=np.int64)
    atom_mask = np.zeros(max_atoms, dtype=bool)
    
    # Fill in actual values
    padded_coords[:n_atoms] = coords
    padded_elements[:n_atoms] = elements
    atom_mask[:n_atoms] = True
    
    return padded_coords, padded_elements, atom_mask


class GSMTrainingDataset(torch.utils.data.Dataset):
    """
    Training dataset for Generative Spectrum-to-Molecule model.
    
    Loads mass spectra with DreaMS embeddings and converts SMILES to 3D coordinates.
    """

    def __init__(
        self,
        data_dir: str,  # Path to exported HDF5 directory
        split: str = "train",
        max_atoms: int = MAX_ATOMS,
        scale: float = 16.0,
        rotation_augment: bool = True,
        use_mmff: bool = True,
        cache_structures: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the GSM training dataset.
        
        Args:
            data_dir: Path to directory containing exported train.h5 and val.h5
            split: "train" or "val"
            max_atoms: Maximum number of atoms to handle
            scale: Coordinate scaling factor
            rotation_augment: Whether to apply random rotation augmentation
            use_mmff: Whether to use MMFF force field
            cache_structures: Whether to cache converted 3D structures
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split
        self.max_atoms = max_atoms
        self.scale = scale
        self.rotation_augment = rotation_augment
        self.use_mmff = use_mmff
        self.cache_structures = cache_structures
        
        # Load exported data (NO DreaMS DEPENDENCY!)
        print(f"Loading {split} data from {data_dir}")
        self.exported_data = ExportedMassSpecGymDataset(
            h5_path=f"{data_dir}/{split}.h5",
            preload_to_memory=True,  # Preload for faster access
        )
        
        print(f"Loaded {len(self.exported_data)} samples for {split} split")
        
        # Cache for converted structures
        self.structure_cache = {} if cache_structures else None

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Get an item from the dataset.
        
        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.
        """
        max_retries = 10
        for retry in range(max_retries):
            # Get SMILES and DreaMS embedding from exported data
            current_idx = (idx + retry) % len(self.exported_data)
            exported_sample = self.exported_data[current_idx]
            smiles = exported_sample['smiles']
            dreams_embedding = exported_sample['dreams_embedding']  # Already a tensor
            data_idx = exported_sample['original_idx']
            
            # Check if we have cached 3D structures
            if 'coords' in exported_sample and 'elements' in exported_sample:
                # Use pre-computed structures from filtered dataset
                coords_tensor = exported_sample['coords']  # [max_atoms, 3]
                elements_str = exported_sample['elements']  # [max_atoms] string array
                n_atoms = int(exported_sample['n_atoms'])
                
                # Extract actual atoms (remove padding)
                coords = coords_tensor[:n_atoms].numpy()
                
                # Convert element symbols to indices
                elements = []
                for elem_str in elements_str[:n_atoms]:
                    if isinstance(elem_str, bytes):
                        elem_str = elem_str.decode('utf-8')
                    elem_str = elem_str.strip()
                    if elem_str and elem_str in ATOM_TO_IDX:
                        elements.append(ATOM_TO_IDX[elem_str])
                    else:
                        # Invalid element, mark as failure
                        elements = None
                        break
                
                if elements is not None:
                    elements = np.array(elements, dtype=np.int64)
                    success = True
                else:
                    success = False
            else:
                # Convert SMILES to 3D coordinates (fallback)
                if self.cache_structures and data_idx in self.structure_cache:
                    coords, elements, success = self.structure_cache[data_idx]
                else:
                    coords, elements, success = smiles_to_3d_coords(
                        smiles,
                        use_mmff=self.use_mmff,
                        random_seed=42 if not self.rotation_augment else None,
                    )
                    
                    if self.cache_structures and success:
                        self.structure_cache[data_idx] = (coords, elements, success)
            
            # If conversion succeeded, break
            if success:
                break
        
        # If all retries failed, use a fallback simple molecule
        if not success:
            print(f"Failed to convert SMILES: {smiles}, trying another sample")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Pad to max_atoms
        coords, elements, atom_mask = pad_atom_features(
            coords, elements, self.max_atoms
        )
        
        # Center coordinates
        n_atoms = atom_mask.sum()
        if n_atoms > 0:
            center = coords[:n_atoms].mean(axis=0)
            coords[:n_atoms] -= center
        
        # Convert to tensors
        coords = torch.from_numpy(coords).float()
        elements = torch.from_numpy(elements).long()
        atom_mask = torch.from_numpy(atom_mask)
        # dreams_embedding is already a tensor from exported_data_loader
        
        # Create one-hot encoding for elements
        element_onehot = torch.nn.functional.one_hot(
            elements, num_classes=NUM_ATOM_TYPES
        ).float()
        
        # Scale coordinates
        coords = coords / self.scale
        
        # Create reference positions (same as coords for now, could be different conformer)
        ref_pos = coords.clone()
        
        # OPTIONAL: Randomly shuffle atom order to enforce permutation invariance during training
        # This ensures the model doesn't learn spurious patterns based on RDKit's atom ordering
        if self.rotation_augment and self.split == "train":
            # Get indices of valid atoms
            valid_indices = torch.where(atom_mask)[0]
            n_valid = len(valid_indices)
            
            # Create random permutation of valid atoms
            perm = torch.randperm(n_valid)
            shuffled_indices = valid_indices[perm]
            
            # Create full index mapping (keep padded atoms at end)
            full_perm = torch.arange(self.max_atoms)
            full_perm[:n_valid] = shuffled_indices
            
            # Apply permutation to all atom features
            coords = coords[full_perm]
            ref_pos = ref_pos[full_perm]
            elements = elements[full_perm]
            element_onehot = element_onehot[full_perm]
            atom_mask = atom_mask[full_perm]
        
        # Optional: Apply random rotation augmentation
        if self.rotation_augment and self.split == "train":
            # Random rotation matrix
            angles = torch.rand(3) * 2 * np.pi
            Rx = torch.tensor([
                [1, 0, 0],
                [0, torch.cos(angles[0]), -torch.sin(angles[0])],
                [0, torch.sin(angles[0]), torch.cos(angles[0])]
            ], dtype=torch.float32)
            Ry = torch.tensor([
                [torch.cos(angles[1]), 0, torch.sin(angles[1])],
                [0, 1, 0],
                [-torch.sin(angles[1]), 0, torch.cos(angles[1])]
            ], dtype=torch.float32)
            Rz = torch.tensor([
                [torch.cos(angles[2]), -torch.sin(angles[2]), 0],
                [torch.sin(angles[2]), torch.cos(angles[2]), 0],
                [0, 0, 1]
            ], dtype=torch.float32)
            R = Rz @ Ry @ Rx
            coords = coords @ R.T
        
        features = {
            "coords": coords.unsqueeze(0),           # [1, max_atoms, 3] - ground truth coordinates
            "ref_pos": ref_pos,                      # [max_atoms, 3] - reference positions
            "ref_element": element_onehot,           # [max_atoms, num_atom_types] - element one-hot
            "atom_pad_mask": atom_mask,              # [max_atoms] - which atoms are valid
            "atom_resolved_mask": atom_mask,         # [max_atoms] - which atoms are resolved
            "dreams_embedding": dreams_embedding,    # [1024] - DreaMS spectrum embedding
            "smiles": smiles,                        # Original SMILES string
            "data_idx": torch.tensor(data_idx, dtype=torch.long),  # Index in full dataset
            "num_atoms": torch.tensor(n_atoms, dtype=torch.long),  # Actual number of atoms
            
            # NOTE: No atom_index or positional encoding features!
            # Molecular structure is permutation-invariant
            # Model should learn spatial relationships from 3D coords + element types only
        }
        
        return features

    def __len__(self) -> int:
        return len(self.exported_data)


def collate(data: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """
    Collate the data.

    Parameters
    ----------
    data : list[dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    dict[str, Tensor]
        The collated data.
    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in ["smiles"]:
            # Check if all have the same shape
            if isinstance(values[0], torch.Tensor):
                shape = values[0].shape
                if all(v.shape == shape for v in values):
                    values = torch.stack(values, dim=0)
                else:
                    # Shouldn't happen with our padding, but handle just in case
                    max_shape = [max(v.shape[i] for v in values) for i in range(len(shape))]
                    padded_values = []
                    for v in values:
                        padding = []
                        for i in range(len(shape) - 1, -1, -1):
                            padding.extend([0, max_shape[i] - v.shape[i]])
                        padded_values.append(torch.nn.functional.pad(v, padding))
                    values = torch.stack(padded_values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class GSMTrainingDataModule(LightningDataModule):
    """DataModule for GSM Training."""

    def __init__(
        self,
        data_dir: str,  # Path to directory with train.h5 and val.h5
        max_atoms: int = MAX_ATOMS,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        scale: float = 16.0,
        rotation_augment_train: bool = True,
        use_mmff: bool = True,
        cache_structures: bool = True,
    ):
        """
        Initialize the GSM training data module.
        
        Args:
            data_dir: Path to directory containing exported train.h5 and val.h5
            max_atoms: Maximum number of atoms
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory
            scale: Coordinate scaling factor
            rotation_augment_train: Whether to augment training data with rotations
            use_mmff: Whether to use MMFF force field
            cache_structures: Whether to cache converted structures
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.max_atoms = max_atoms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.scale = scale
        self.rotation_augment_train = rotation_augment_train
        self.use_mmff = use_mmff
        self.cache_structures = cache_structures
        
        self.batch_size_per_device = batch_size
        self.batch_size_per_device_test = batch_size  # Use same batch size for val

        # Create train dataset
        self._train_set = GSMTrainingDataset(
            data_dir=data_dir,
            split="train",
            max_atoms=max_atoms,
            scale=scale,
            rotation_augment=rotation_augment_train,
            use_mmff=use_mmff,
            cache_structures=cache_structures,
        )
        
        # Create validation dataset
        self._val_set = GSMTrainingDataset(
            data_dir=data_dir,
            split="val",
            max_atoms=max_atoms,
            scale=scale,
            rotation_augment=False,  # No augmentation for validation
            use_mmff=use_mmff,
            cache_structures=cache_structures,
        )
        
        print(f"Training dataset size: {len(self._train_set)}")
        print(f"Validation dataset size: {len(self._val_set)}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Run the setup for the DataModule.

        Parameters
        ----------
        stage : str, optional
            The stage, one of 'fit', 'validate', 'test'.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.batch_size // self.trainer.world_size
            )

    def train_dataloader(self) -> DataLoader:
        """
        Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.
        """
        return DataLoader(
            self._train_set,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=collate,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Get the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader.
        """
        return DataLoader(
            self._val_set,
            batch_size=self.batch_size_per_device_test,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate,
            drop_last=False,
        )


if __name__ == "__main__":
    # Test the dataset
    import sys
    
    # Path to exported data directory
    data_dir = "../data/massspecgym_exported"
    
    print("Testing GSMTrainingDataset...")
    dataset = GSMTrainingDataset(
        data_dir=data_dir,
        split="train",
        max_atoms=MAX_ATOMS,
        cache_structures=False,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a sample
    print("\nLoading sample 0...")
    sample = dataset[0]
    
    print("\nSample keys:", sample.keys())
    print(f"SMILES: {sample['smiles']}")
    print(f"Coords shape: {sample['coords'].shape}")
    print(f"Ref pos shape: {sample['ref_pos'].shape}")
    print(f"Ref pos: {sample['ref_pos']}")
    print(f"Elements shape: {sample['ref_element'].shape}")
    print(f"Atom mask shape: {sample['atom_pad_mask'].shape}")
    print(f"DreaMS embedding shape: {sample['dreams_embedding'].shape}")
    print(f"Number of atoms: {sample['num_atoms']}")
    
    # Test batch collation
    print("\nTesting batch collation...")
    batch = collate([dataset[i] for i in range(4)])
    print(f"Batch coords shape: {batch['coords'].shape}")
    print(f"Batch DreaMS embeddings shape: {batch['dreams_embedding'].shape}")
    
    print("\n✓ All tests passed!")
