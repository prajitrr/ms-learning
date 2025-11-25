"""
Updated train_datamodule.py that uses exported HDF5 data.

This version works in SimpleFold environment without DreaMS dependency.
"""

# Just show the key changes needed to integrate exported data

# CHANGE 1: Import the exported data loader
from datasets.exported_data_loader import load_exported_data

# CHANGE 2: Update GSMTrainingDataset.__init__
class GSMTrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,  # Changed from data_path + fold_path
        split: str = "train",
        max_atoms: int = MAX_ATOMS,
        scale: float = 16.0,
        rotation_augment: bool = True,
        use_mmff: bool = True,
        cache_structures: bool = True,
        **kwargs,
    ) -> None:
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
        self.exported_data = load_exported_data(
            data_dir=data_dir,
            split=split,
            preload=True,  # Preload to memory for speed
        )
        
        print(f"Loaded {len(self.exported_data)} samples for {split} split")
        
        # Cache for converted structures
        self.structure_cache = {} if cache_structures else None

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # Get SMILES and DreaMS embedding from exported data
        exported_sample = self.exported_data[idx]
        smiles = exported_sample['smiles']
        dreams_embedding = exported_sample['dreams_embedding']  # Already a tensor!
        data_idx = exported_sample['original_idx']
        
        # Convert SMILES to 3D coordinates (rest is the same)
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
        
        # ... rest of __getitem__ is exactly the same ...
        
    def __len__(self) -> int:
        return len(self.exported_data)


# CHANGE 3: Update GSMTrainingDataModule.__init__
class GSMTrainingDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,  # Changed: single directory path
        max_atoms: int = MAX_ATOMS,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        scale: float = 16.0,
        rotation_augment_train: bool = True,
        use_mmff: bool = True,
        cache_structures: bool = True,
    ):
        super().__init__()
        
        self.save_hyperparameters(logger=False)
        
        self.data_dir = data_dir
        # ... rest of params same ...
        
        # Create train dataset
        self._train_set = GSMTrainingDataset(
            data_dir=data_dir,  # Single path to exported directory
            split="train",
            max_atoms=max_atoms,
            scale=scale,
            rotation_augment=rotation_augment_train,
            use_mmff=use_mmff,
            cache_structures=cache_structures,
        )
        
        # Create validation dataset
        self._val_set = GSMTrainingDataset(
            data_dir=data_dir,  # Same directory, different split
            split="val",
            max_atoms=max_atoms,
            scale=scale,
            rotation_augment=False,
            use_mmff=use_mmff,
            cache_structures=cache_structures,
        )


# USAGE:
if __name__ == "__main__":
    # NEW: Point to exported data directory
    data_dir = "../data/massspecgym_exported"
    
    # Initialize datamodule (much simpler!)
    datamodule = GSMTrainingDataModule(
        data_dir=data_dir,
        max_atoms=80,
        batch_size=32,
        num_workers=4,
    )
    
    # Setup
    datamodule.setup()
    
    # Get dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    # Test loading
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Coords shape: {batch['coords'].shape}")
    print(f"DreaMS embeddings shape: {batch['dreams_embedding'].shape}")
