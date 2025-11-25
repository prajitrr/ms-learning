#
# Lightning Module for Molecular Structure Generation
#

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.flow import LinearPath


class MolecularFlowMatching(pl.LightningModule):
    """
    Lightning module for training molecular structure generation with flow matching.
    """
    
    def __init__(
        self,
        model,
        processor,
        lr=1e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        max_steps=100000,
        t_eps=0.001,  # Minimum timestep (avoid t=0)
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'processor'])
        
        self.model = model
        self.processor = processor
        self.path = LinearPath()
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.t_eps = t_eps
    
    def forward(self, noised_pos, t, feats):
        return self.model(noised_pos, t, feats)
    
    def compute_loss(self, batch):
        """
        Compute flow matching loss.
        
        Loss = MSE(predicted_velocity, target_velocity) with masking
        """
        # Get ground truth coordinates
        coords_gt = batch['coords']  # [B, N, 3]
        atom_mask = batch['atom_pad_mask']  # [B, N]
        
        B, N, _ = coords_gt.shape
        
        # Sample timesteps (avoid t=0 with t_eps)
        t = torch.rand(B, device=self.device) * (1 - self.t_eps) + self.t_eps
        
        # Sample noise
        noise = torch.randn_like(coords_gt)
        
        # Compute interpolated sample and target velocity
        # x0 = noise, x1 = ground truth
        t_expanded, noised_coords, target_velocity = self.path.interpolant(t, noise, coords_gt)
        
        # Forward pass: predict velocity
        output = self.forward(noised_coords, t, batch)
        pred_velocity = output['predict_velocity']  # [B, N, 3]
        
        # Compute loss with masking
        loss = F.mse_loss(pred_velocity, target_velocity, reduction='none')  # [B, N, 3]
        
        # Apply atom mask
        loss = loss * atom_mask.unsqueeze(-1).float()
        
        # Average over valid atoms
        loss = loss.sum() / (atom_mask.sum() * 3 + 1e-8)
        
        return loss, {
            'noised_coords': noised_coords,
            'pred_velocity': pred_velocity,
            'target_velocity': target_velocity,
        }
    
    def training_step(self, batch, batch_idx):
        # Preprocess batch
        batch = self.processor.preprocess_training(batch)
        
        # Compute loss
        loss, outputs = self.compute_loss(batch)
        
        # Log
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Preprocess batch
        batch = self.processor.preprocess_inference(batch)
        
        # Compute loss
        loss, outputs = self.compute_loss(batch)
        
        # Log
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_steps,
            eta_min=1e-6,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }


if __name__ == "__main__":
    print("Testing MolecularFlowMatching...")
    print("=" * 80)
    
    from model.model_factory import create_molecular_dit
    from molecular_data_processor import MolecularDataProcessor
    
    # Create model
    model = create_molecular_dit(
        hidden_size=256,
        num_heads=4,
        depth=4,
    )
    
    # Create processor
    processor = MolecularDataProcessor(
        device=torch.device('cpu'),
        scale=16.0,
        multiplicity=1,
    )
    
    # Create Lightning module
    pl_module = MolecularFlowMatching(
        model=model,
        processor=processor,
        lr=1e-4,
    )
    
    print(f"✓ Module created successfully!")
    print(f"  Parameters: {sum(p.numel() for p in pl_module.parameters()):,}")
    
    # Create fake batch
    batch = {
        'coords': torch.randn(2, 80, 3),
        'ref_pos': torch.randn(2, 80, 3),
        'ref_element': torch.randn(2, 80, 13),
        'atom_pad_mask': torch.ones(2, 80).bool(),
        'atom_resolved_mask': torch.ones(2, 80).bool(),
        'dreams_embedding': torch.randn(2, 1024),
        'num_atoms': torch.tensor([45, 62]),
    }
    
    # Test training step
    loss = pl_module.training_step(batch, 0)
    print(f"\n✓ Training step successful!")
    print(f"  Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
