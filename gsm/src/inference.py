"""
Inference script for Molecular Structure Generation from Mass Spectra.

Generates 3D molecular structures from mass spectra embeddings using flow matching.
"""

import os
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import lightning.pytorch as pl

from model.model_factory import create_molecular_dit
from model.flow import LinearPath
from molecular_data_processor import MolecularDataProcessor
from datasets.exported_data_loader import ExportedMassSpecGymDataset
from refine_molecule import MoleculeRefiner


class EulerSampler:
    """
    Euler ODE sampler for flow matching.
    
    Simpler than SDE sampler - just follows the learned velocity field.
    """
    
    def __init__(
        self,
        num_timesteps=100,
        t_start=1e-4,
        t_end=1.0,
    ):
        self.num_timesteps = num_timesteps
        self.t_start = t_start
        self.t_end = t_end
        
        # Create timestep schedule (linear or log-spaced)
        self.steps = torch.linspace(t_start, t_end, num_timesteps + 1)
    
    @torch.no_grad()
    def euler_step(
        self,
        model,
        y,
        t,
        dt,
        batch,
    ):
        """
        Single Euler ODE step.
        
        dy/dt = velocity(y, t)
        y_{t+dt} = y_t + velocity(y_t, t) * dt
        """
        # Expand timestep to batch dimension
        batched_t = t.expand(y.shape[0])
        
        # Predict velocity
        output = model(
            noised_pos=y,
            t=batched_t,
            feats=batch,
        )
        velocity = output['predict_velocity']
        
        # Euler step
        y_next = y + velocity * dt
        
        return y_next
    
    @torch.no_grad()
    def sample(self, model, noise, batch, show_progress=True):
        """
        Sample from noise to data by integrating the ODE.
        
        Args:
            model: The trained diffusion model
            noise: Initial noise [B, N, 3]
            batch: Batch dictionary with features
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with 'denoised_coords' [B, N, 3]
        """
        steps = self.steps.to(noise.device)
        y = noise
        
        iterator = range(self.num_timesteps)
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling", total=self.num_timesteps)
        
        for i in iterator:
            t = steps[i]
            t_next = steps[i + 1]
            dt = t_next - t
            
            y = self.euler_step(model, y, t, dt, batch)
        
        return {
            "denoised_coords": y
        }


def load_model(checkpoint_path, device, **model_kwargs):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .ckpt file
        device: Device to load model on
        **model_kwargs: Model architecture arguments
        
    Returns:
        Loaded model in eval mode
    """
    # Create model
    model = create_molecular_dit(**model_kwargs)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present (from Lightning)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def prepare_batch_from_dataset(dataset, indices, processor, device, model_max_atoms):
    """
    Prepare a batch from dataset indices.
    
    Args:
        dataset: ExportedMassSpecGymDataset
        indices: List of sample indices
        processor: MolecularDataProcessor
        device: Device
        model_max_atoms: Max atoms expected by model (for padding)
        
    Returns:
        Batch dictionary ready for model
    """
    # Get samples
    samples = [dataset[idx] for idx in indices]
    
    # Stack into batch
    batch = {
        'smiles': [s['smiles'] for s in samples],
        'dreams_embedding': torch.stack([s['dreams_embedding'] for s in samples]),
        'original_idx': torch.tensor([s['original_idx'] for s in samples]),
    }
    
    # If we have cached structures, load them
    if 'coords' in samples[0]:
        coords_list = [s['coords'] for s in samples]
        elements_list = [s['elements'] for s in samples]
        n_atoms_list = [s['n_atoms'] for s in samples]
        
        # Get dataset max_atoms
        dataset_max_atoms = coords_list[0].shape[0]
        
        # Convert element strings to one-hot (using GSM atom types)
        from datasets.train_datamodule import ATOM_TO_IDX, NUM_ATOM_TYPES
        
        element_onehot_list = []
        atom_mask_list = []
        coords_padded_list = []
        ref_pos_padded_list = []
        
        for i, elements in enumerate(elements_list):
            n_atoms = int(n_atoms_list[i])
            
            # Convert element symbols to indices
            element_indices = torch.zeros(dataset_max_atoms, dtype=torch.long)
            for j, elem_str in enumerate(elements[:n_atoms]):
                if isinstance(elem_str, bytes):
                    elem_str = elem_str.decode('utf-8')
                elem_str = elem_str.strip()
                if elem_str in ATOM_TO_IDX:
                    element_indices[j] = ATOM_TO_IDX[elem_str]
            
            # Pad to model_max_atoms if needed
            if dataset_max_atoms < model_max_atoms:
                # Pad element indices
                element_indices_padded = torch.zeros(model_max_atoms, dtype=torch.long)
                element_indices_padded[:dataset_max_atoms] = element_indices
                element_indices = element_indices_padded
                
                # Pad coords
                coords_padded = torch.zeros(model_max_atoms, 3)
                coords_padded[:dataset_max_atoms] = coords_list[i]
                coords_padded_list.append(coords_padded)
                
                # Pad ref_pos
                ref_pos_padded = torch.zeros(model_max_atoms, 3)
                ref_pos_padded[:dataset_max_atoms] = coords_list[i]
                ref_pos_padded_list.append(ref_pos_padded)
            else:
                coords_padded_list.append(coords_list[i][:model_max_atoms])
                ref_pos_padded_list.append(coords_list[i][:model_max_atoms])
            
            # One-hot encode
            element_onehot = torch.nn.functional.one_hot(
                element_indices, num_classes=NUM_ATOM_TYPES
            ).float()
            element_onehot_list.append(element_onehot)
            
            # Create mask
            mask = torch.zeros(model_max_atoms, dtype=torch.bool)
            mask[:n_atoms] = True
            atom_mask_list.append(mask)
        
        batch['coords'] = torch.stack(coords_padded_list)
        batch['ref_pos'] = torch.stack(ref_pos_padded_list)
        batch['ref_element'] = torch.stack(element_onehot_list)
        batch['atom_pad_mask'] = torch.stack(atom_mask_list)
        batch['atom_resolved_mask'] = batch['atom_pad_mask'].clone()
        batch['num_atoms'] = torch.tensor(n_atoms_list, dtype=torch.long)
        
        # Scale coordinates (dataset stores unscaled)
        batch['coords'] = batch['coords'] / processor.scale
        batch['ref_pos'] = batch['ref_pos'] / processor.scale
    
    # Move to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    
    return batch


def run_inference(args):
    """
    Main inference function.
    """
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\n[1/5] Loading model...")
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        depth=args.depth,
        max_atoms=args.max_atoms,
        num_atom_types=13,
        dreams_embedding_dim=1024,
        pos_embed_dim=258,
        atom_n_queries=args.atom_n_queries,
        atom_n_keys=args.atom_n_keys,
        dreams_dropout_prob=0.0,  # No dropout during inference
        use_length_condition=True,
        use_swiglu=True,
    )
    
    # Create processor
    print("\n[2/5] Creating data processor...")
    processor = MolecularDataProcessor(
        device=device,
        scale=16.0,
        multiplicity=1,
        inference_multiplicity=1,
        backend="torch",
    )
    
    # Create sampler
    print("\n[3/5] Creating sampler...")
    sampler = EulerSampler(
        num_timesteps=args.num_steps,
        t_start=args.t_start,
        t_end=1.0,
    )
    
    # Create molecule refiner
    print("\n[4/5] Creating molecule refiner...")
    refiner = MoleculeRefiner(
        force_field=args.force_field,
        max_iters=args.refine_iters,
        verbose=args.verbose,
    )
    
    # Load dataset
    print("\n[5/5] Loading data...")
    dataset = ExportedMassSpecGymDataset(
        h5_path=args.data_path,
        preload_to_memory=True,
    )
    print(f"  Dataset size: {len(dataset)}")
    
    # Determine samples to process
    if args.num_samples is not None:
        num_samples = min(args.num_samples, len(dataset))
    else:
        num_samples = len(dataset)
    
    sample_indices = list(range(args.start_idx, args.start_idx + num_samples))
    
    print(f"\n{'='*80}")
    print(f"Generating structures for {num_samples} samples...")
    print(f"{'='*80}\n")
    
    # Process in batches
    results = []
    
    for batch_start in tqdm(range(0, len(sample_indices), args.batch_size), desc="Batches"):
        batch_indices = sample_indices[batch_start:batch_start + args.batch_size]
        
        # Prepare batch (with padding to model's max_atoms)
        batch = prepare_batch_from_dataset(dataset, batch_indices, processor, device, args.max_atoms)
        
        # Generate multiple samples per input if requested
        all_coords = []
        for sample_idx in range(args.num_samples_per_spectrum):
            # Sample noise
            noise_shape = (len(batch_indices), args.max_atoms, 3)
            noise = torch.randn(noise_shape, device=device)
            
            # Run inference
            with torch.no_grad():
                output = sampler.sample(
                    model=model,
                    noise=noise,
                    batch=batch,
                    show_progress=(batch_start == 0 and sample_idx == 0),  # Only show first
                )
            
            coords = output['denoised_coords']
            all_coords.append(coords)
        
        # Stack samples [B, num_samples, N, 3]
        all_coords = torch.stack(all_coords, dim=1)
        
        # Post-process and refine
        for i, idx in enumerate(batch_indices):
            sample = dataset[idx]
            smiles_gt = sample['smiles']
            
            for sample_idx in range(args.num_samples_per_spectrum):
                coords = all_coords[i, sample_idx]  # [N, 3]
                mask = batch['atom_pad_mask'][i]  # [N]
                elements = batch['ref_element'][i]  # [N, num_atom_types]
                n_atoms = int(batch['num_atoms'][i])
                
                # Rescale coordinates
                coords = coords * processor.scale
                
                # Extract valid atoms
                coords_valid = coords[:n_atoms].cpu().numpy()
                elements_valid = elements[:n_atoms].argmax(dim=-1).cpu().numpy()
                
                # Convert element indices to symbols
                from datasets.train_datamodule import IDX_TO_ATOM
                atom_symbols = [IDX_TO_ATOM[int(e)] for e in elements_valid]
                
                # Refine to valid molecule
                if args.refine:
                    mol, refined_coords, success = refiner.refine_to_molecule(
                        coords=coords_valid,
                        elements=atom_symbols,
                        guess_bonds=args.guess_bonds,
                    )
                else:
                    mol = None
                    refined_coords = coords_valid
                    success = False
                
                result = {
                    'idx': idx,
                    'sample_idx': sample_idx,
                    'smiles_gt': smiles_gt,
                    'coords_raw': coords_valid,
                    'coords_refined': refined_coords,
                    'elements': atom_symbols,
                    'n_atoms': n_atoms,
                    'mol': mol,
                    'refinement_success': success,
                }
                
                results.append(result)
    
    # Save results
    print(f"\n{'='*80}")
    print(f"Saving results...")
    print(f"{'='*80}\n")
    
    save_results(results, output_dir, args)
    
    # Print statistics
    if args.refine:
        num_refined = sum(r['refinement_success'] for r in results)
        print(f"\nRefinement statistics:")
        print(f"  Total structures: {len(results)}")
        print(f"  Successfully refined: {num_refined} ({100*num_refined/len(results):.1f}%)")
    
    print(f"\n✓ Results saved to {output_dir}")


def save_results(results, output_dir, args):
    """
    Save inference results.
    """
    import json
    import pickle
    from rdkit import Chem
    
    # Save summary JSON
    summary = []
    for r in results:
        summary.append({
            'idx': r['idx'],
            'sample_idx': r['sample_idx'],
            'smiles_gt': r['smiles_gt'],
            'n_atoms': r['n_atoms'],
            'refinement_success': r['refinement_success'],
        })
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save full results as pickle
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save individual structures
    structures_dir = output_dir / 'structures'
    structures_dir.mkdir(exist_ok=True)
    
    for r in results:
        base_name = f"sample_{r['idx']:05d}_gen_{r['sample_idx']}"
        
        # Save coordinates as XYZ
        xyz_path = structures_dir / f"{base_name}.xyz"
        save_xyz(r['coords_refined'], r['elements'], xyz_path)
        
        # Save molecule as SDF if refinement succeeded
        if r['refinement_success'] and r['mol'] is not None:
            sdf_path = structures_dir / f"{base_name}.sdf"
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(r['mol'])
            writer.close()


def save_xyz(coords, elements, path):
    """
    Save structure as XYZ file.
    """
    with open(path, 'w') as f:
        f.write(f"{len(coords)}\n")
        f.write(f"Generated molecular structure\n")
        for elem, (x, y, z) in zip(elements, coords):
            f.write(f"{elem} {x:.6f} {y:.6f} {z:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Molecular structure generation from mass spectra")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--max_atoms", type=int, default=80)
    parser.add_argument("--atom_n_queries", type=int, default=None, help="Local attention queries (None=full)")
    parser.add_argument("--atom_n_keys", type=int, default=None, help="Local attention keys")
    
    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to HDF5 file (val.h5)")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    # Sampling
    parser.add_argument("--num_steps", type=int, default=100, help="Number of ODE integration steps")
    parser.add_argument("--t_start", type=float, default=1e-4, help="Starting timestep")
    parser.add_argument("--num_samples_per_spectrum", type=int, default=1, help="Samples per spectrum")
    
    # Refinement
    parser.add_argument("--refine", action="store_true", help="Refine to valid molecules")
    parser.add_argument("--guess_bonds", action="store_true", help="Guess bonds from distances")
    parser.add_argument("--force_field", type=str, default="MMFF", choices=["MMFF", "UFF"], help="Force field for refinement")
    parser.add_argument("--refine_iters", type=int, default=500, help="Refinement iterations")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    run_inference(args)


if __name__ == "__main__":
    main()
