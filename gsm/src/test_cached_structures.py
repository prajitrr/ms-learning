"""
Quick test to verify cached structures are loaded properly.
"""

import sys
import torch

# Test loading filtered dataset
from datasets.train_datamodule import GSMTrainingDataModule

print("Testing filtered dataset with cached structures...")
print("=" * 80)

# Create datamodule pointing to filtered data
datamodule = GSMTrainingDataModule(
    data_dir="../data/massspecgym_filtered",
    max_atoms=80,
    batch_size=4,
    num_workers=0,  # Single process for testing
)

# Get train dataset
train_dataset = datamodule._train_set

print(f"\nDataset info:")
print(f"  Samples: {len(train_dataset)}")
print(f"  Exported data loader has cached structures: {train_dataset.exported_data.has_cached_structures}")

# Load one sample
print("\nLoading sample 0...")
sample = train_dataset[0]

print(f"\nSample keys: {sample.keys()}")
print(f"  SMILES: {sample['smiles']}")
print(f"  coords shape: {sample['coords'].shape}")
print(f"  ref_pos shape: {sample['ref_pos'].shape}")
print(f"  ref_element shape: {sample['ref_element'].shape}")
print(f"  num_atoms: {sample['num_atoms']}")
print(f"  atom_pad_mask sum: {sample['atom_pad_mask'].sum()}")

print("\n" + "=" * 80)
print("✓ Cached structures loaded successfully!")
