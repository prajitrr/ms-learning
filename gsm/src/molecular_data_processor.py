import torch
import numpy as np
from scipy.spatial.transform import Rotation


class MolecularDataProcessor:
    def __init__(
        self,
        device,
        scale=16.0,
        ref_scale=5.0,
        multiplicity=1,
        inference_multiplicity=1,
        backend="torch",
    ):
        """
        Data processor for molecular structures.
        
        Compatible with GSMTrainingDataset which already handles SMILES → 3D conversion.
        This processor handles batching, scaling, augmentation, and device placement.
        
        Args:
            device: torch device
            scale: scaling factor for coordinates (should match dataset)
            ref_scale: scaling factor for reference positions
            multiplicity: training batch multiplicity (for data augmentation)
            inference_multiplicity: inference batch multiplicity
            backend: 'torch' or 'mlx'
        """
        self.device = device
        self.scale = scale
        self.ref_scale = ref_scale
        self.multiplicity = multiplicity
        self.inference_multiplicity = inference_multiplicity
        self.backend = backend

    def center_coords(self, coords, mask=None):
        """
        Center coordinates around their center of mass.
        
        Args:
            coords: (B, N, 3) tensor of coordinates
            mask: (B, N) boolean tensor indicating valid atoms
            
        Returns:
            centered_coords: (B, N, 3) tensor of centered coordinates
        """
        if mask is None:
            center = coords.mean(dim=1, keepdim=True)
        else:
            # Only use non-masked atoms for centering
            mask_expanded = mask.unsqueeze(-1).float()
            coords_sum = (coords * mask_expanded).sum(dim=1, keepdim=True)
            coords_count = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1)
            center = coords_sum / coords_count
        
        return coords - center

    def random_rotation_matrix(self, batch_size=1):
        """
        Generate random rotation matrices using scipy.
        
        Args:
            batch_size: number of rotation matrices to generate
            
        Returns:
            rotation_matrices: (B, 3, 3) tensor of rotation matrices
        """
        rotation_matrices = []
        for _ in range(batch_size):
            rot = Rotation.random()
            rotation_matrices.append(rot.as_matrix())
        
        rotation_matrices = np.stack(rotation_matrices, axis=0)
        return torch.from_numpy(rotation_matrices).float()

    def apply_random_rotation(self, coords, mask=None):
        """
        Apply random rotation to coordinates.
        
        Args:
            coords: (B, N, 3) tensor of coordinates
            mask: (B, N) boolean tensor indicating valid atoms
            
        Returns:
            rotated_coords: (B, N, 3) tensor of rotated coordinates
        """
        B = coords.shape[0]
        
        # Generate random rotation matrices
        rot_matrices = self.random_rotation_matrix(batch_size=B).to(coords.device)
        
        # Apply rotation: coords @ R^T
        rotated_coords = torch.matmul(coords, rot_matrices.transpose(-2, -1))
        
        return rotated_coords

    def center_random_augmentation(self, coords, mask, centering=True, augmentation=True):
        """
        Center and optionally apply random rotation to coordinates.
        
        Args:
            coords: (B, N, 3) tensor of coordinates
            mask: (B, N) boolean tensor indicating valid atoms
            centering: whether to center coordinates
            augmentation: whether to apply random rotation
            
        Returns:
            processed_coords: (B, N, 3) tensor of processed coordinates
        """
        if centering:
            coords = self.center_coords(coords, mask)
        
        if augmentation:
            coords = self.apply_random_rotation(coords, mask)
        
        return coords

    def batch_to_device(self, batch, multiplicity=1):
        """
        Move batch tensors to device and optionally repeat for multiplicity.
        
        Args:
            batch: dictionary of tensors
            multiplicity: number of times to repeat each sample
            
        Returns:
            batch: processed batch on device
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if multiplicity > 1:
                    v = v.repeat_interleave(multiplicity, dim=0)
                batch[k] = v.to(self.device)
        return batch

    def preprocess_training(self, batch):
        """
        Preprocess batch for training.
        
        NOTE: Assumes batch already contains 'coords', 'ref_pos', 'ref_element', etc.
        from GSMTrainingDataset. This processor handles scaling, device placement,
        and multiplicity augmentation.
        
        Args:
            batch: dictionary from GSMTrainingDataset containing:
                - coords: [B, 1, N, 3] ground truth coordinates (already scaled by dataset)
                - ref_pos: [B, N, 3] reference positions (already scaled by dataset)
                - ref_element: [B, N, num_elements] element one-hot
                - atom_pad_mask: [B, N] valid atoms mask
                - dreams_embedding: [B, 1024] DreaMS embeddings
                - etc.
            
        Returns:
            batch: processed batch ready for model
        """
        # Coords are already scaled by dataset (divided by self.scale)
        # Just squeeze the time dimension added by dataset
        if batch['coords'].dim() == 4:
            batch['coords'] = batch['coords'].squeeze(1)  # [B, 1, N, 3] -> [B, N, 3]
        
        # Move to device with multiplicity
        batch = self.batch_to_device(batch, multiplicity=self.multiplicity)
        
        # Apply random augmentation if multiplicity > 1
        # (dataset already does rotation, but we can apply another for extra augmentation)
        if self.multiplicity > 1:
            batch['coords'] = self.center_random_augmentation(
                batch['coords'],
                batch['atom_pad_mask'],
                centering=True,
                augmentation=True,
            )
        
        return batch

    def preprocess_inference(self, batch):
        """
        Preprocess batch for inference.
        
        Args:
            batch: dictionary from GSMTrainingDataset
            
        Returns:
            batch: processed batch ready for inference
        """
        # Squeeze time dimension if present
        if batch['coords'].dim() == 4:
            batch['coords'] = batch['coords'].squeeze(1)  # [B, 1, N, 3] -> [B, N, 3]
        
        # Move to device (no augmentation for inference)
        batch = self.batch_to_device(batch, multiplicity=self.inference_multiplicity)
        
        return batch

    def postprocess(self, out_dict, batch):
        """
        Postprocess model outputs.
        
        Args:
            out_dict: dictionary containing model outputs
            batch: input batch
            
        Returns:
            out_dict: processed outputs with rescaled coordinates
        """
        out_dict['coords'] = self.center_random_augmentation(
            batch['coords'],
            batch['atom_pad_mask'],
            centering=True,
            augmentation=False,
        ) * self.scale
        
        out_dict['denoised_coords'] = self.center_random_augmentation(
            out_dict['denoised_coords'],
            batch['atom_pad_mask'],
            centering=True,
            augmentation=False,
        ) * self.scale
        
        return out_dict


if __name__ == "__main__":
    # Test the processor with sample data
    print("Testing MolecularDataProcessor...")
    print("=" * 80)
    
    # Create processor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = MolecularDataProcessor(
        device=device,
        scale=16.0,
        multiplicity=4,
    )
    
    print(f"Device: {device}")
    print(f"Scale: {processor.scale}")
    print(f"Multiplicity: {processor.multiplicity}")
    
    # Create fake batch (simulating output from GSMTrainingDataset)
    batch_size = 2
    max_atoms = 80
    
    fake_batch = {
        'coords': torch.randn(batch_size, 1, max_atoms, 3),  # [B, 1, N, 3]
        'ref_pos': torch.randn(batch_size, max_atoms, 3),    # [B, N, 3]
        'ref_element': torch.randn(batch_size, max_atoms, 13),  # [B, N, 13]
        'atom_pad_mask': torch.ones(batch_size, max_atoms).bool(),
        'atom_resolved_mask': torch.ones(batch_size, max_atoms).bool(),
        'dreams_embedding': torch.randn(batch_size, 1024),
        'smiles': ['CCO', 'C1=CC=CC=C1'],
        'num_atoms': torch.tensor([10, 12]),
    }
    
    print(f"\nOriginal batch shapes:")
    for k, v in fake_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    
    # Test preprocessing for training
    print(f"\nPreprocessing for training (multiplicity={processor.multiplicity})...")
    processed_batch = processor.preprocess_training(fake_batch.copy())
    
    print(f"\nProcessed batch shapes:")
    for k, v in processed_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
            print(f"    - Device: {v.device}")
    
    # Verify multiplicity worked
    expected_batch_size = batch_size * processor.multiplicity
    assert processed_batch['coords'].shape[0] == expected_batch_size, \
        f"Expected batch size {expected_batch_size}, got {processed_batch['coords'].shape[0]}"
    
    # Verify coords shape is correct
    assert processed_batch['coords'].shape == (expected_batch_size, max_atoms, 3), \
        f"Expected coords shape ({expected_batch_size}, {max_atoms}, 3), got {processed_batch['coords'].shape}"
    
    print(f"\n✓ Training preprocessing test passed!")
    
    # Test preprocessing for inference
    print(f"\nPreprocessing for inference...")
    inference_batch = processor.preprocess_inference(fake_batch.copy())
    
    print(f"\nInference batch shapes:")
    for k, v in inference_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    
    assert inference_batch['coords'].shape[0] == batch_size, \
        f"Expected batch size {batch_size}, got {inference_batch['coords'].shape[0]}"
    
    print(f"\n✓ Inference preprocessing test passed!")
    
    # Test postprocessing
    print(f"\nTesting postprocessing...")
    fake_output = {
        'denoised_coords': torch.randn(batch_size, max_atoms, 3).to(device),
    }
    
    postprocessed = processor.postprocess(fake_output, inference_batch)
    
    print(f"Postprocessed shapes:")
    for k, v in postprocessed.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    
    print(f"\n✓ Postprocessing test passed!")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)

