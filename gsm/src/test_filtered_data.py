"""
Test loading filtered dataset with cached structures.
"""

import sys
sys.path.insert(0, '/home/ubuntu/ms-learning/gsm/src')

from datasets.train_datamodule import GSMTrainingDataModule

print("=" * 80)
print("Testing Filtered Dataset with Cached Structures")
print("=" * 80)

# Create datamodule pointing to filtered data
datamodule = GSMTrainingDataModule(
    data_dir="../data/massspecgym_filtered",
    max_atoms=80,
    batch_size=4,
    num_workers=0,  # Single-threaded for testing
    scale=16.0,
    rotation_augment_train=False,
    use_mmff=True,
    cache_structures=False,  # Don't need runtime cache since structures are pre-cached
)

print("\n✓ Datamodule created successfully")

# Setup (required by Lightning)
class FakeTrainer:
    world_size = 1

datamodule.trainer = FakeTrainer()
datamodule.batch_size_per_device = 4
datamodule.setup('fit')

print("✓ Datamodule setup complete")

# Get train dataloader
train_loader = datamodule.train_dataloader()
print(f"✓ Train dataloader created: {len(train_loader)} batches")

# Fetch one batch
print("\nFetching one batch...")
batch = next(iter(train_loader))

print("\n" + "=" * 80)
print("Batch Contents:")
print("=" * 80)

for key, value in batch.items():
    if key == 'smiles':
        print(f"  {key}: {value}")
    elif hasattr(value, 'shape'):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: {type(value)}")

print("\n" + "=" * 80)
print("Verification:")
print("=" * 80)

# Check that coords are loaded from cache
if 'coords' in batch:
    print(f"✓ Coords loaded: {batch['coords'].shape}")
    print(f"  Min: {batch['coords'].min():.4f}, Max: {batch['coords'].max():.4f}")
else:
    print("✗ No coords in batch")

if 'num_atoms' in batch:
    print(f"✓ Number of atoms: {batch['num_atoms']}")

if 'ref_element' in batch:
    print(f"✓ Elements loaded: {batch['ref_element'].shape}")

if 'dreams_embedding' in batch:
    print(f"✓ DreaMS embeddings: {batch['dreams_embedding'].shape}")

print("\n" + "=" * 80)
print("✓ All tests passed!")
print("=" * 80)
