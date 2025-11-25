#
# Training script for Molecular Structure Generation
#

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from datasets.train_datamodule import GSMTrainingDataModule
from molecular_data_processor import MolecularDataProcessor
from model.model_factory import create_molecular_dit
from train_module import MolecularFlowMatching


def train(
    # Data
    data_dir="../data/massspecgym_filtered",
    max_atoms=80,
    batch_size=32,
    num_workers=4,
    # Model
    hidden_size=512,
    num_heads=8,
    depth=12,
    mlp_ratio=4.0,
    dreams_dropout_prob=0.1,
    # Training
    lr=1e-4,
    weight_decay=0.01,
    max_epochs=None,  # If set, overrides max_steps
    max_steps=100000,
    warmup_steps=1000,
    # System
    devices=1,
    precision="bf16-mixed",
    gradient_clip_val=1.0,
    accumulate_grad_batches=1,
    # Logging
    project_name="molecular-generation",
    experiment_name="molecular-dit",
    log_every_n_steps=10,
    val_check_interval=1000,
    save_top_k=3,
    # Resume
    resume_from_checkpoint=None,
    wandb_id=None,
):
    """
    Train molecular structure generation model.
    """
    
    pl.seed_everything(42)
    
    print("=" * 80)
    print("Molecular Structure Generation Training")
    print("=" * 80)
    
    # Create datamodule
    print("\n[1/5] Creating datamodule...")
    datamodule = GSMTrainingDataModule(
        data_dir=data_dir,
        max_atoms=max_atoms,
        batch_size=batch_size,
        num_workers=num_workers,
        scale=16.0,
        rotation_augment_train=True,
        use_mmff=True,
        cache_structures=True,
    )
    
    # Create processor
    print("[2/5] Creating data processor...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = MolecularDataProcessor(
        device=device,
        scale=16.0,
        multiplicity=4,  # Data augmentation via multiplicity
    )
    
    # Create model
    print("[3/5] Creating model...")
    model = create_molecular_dit(
        hidden_size=hidden_size,
        num_heads=num_heads,
        depth=depth,
        mlp_ratio=mlp_ratio,
        max_atoms=max_atoms,
        num_atom_types=13,
        dreams_embedding_dim=1024,
        pos_embed_dim=258,
        atom_n_queries=None,  # Full attention
        atom_n_keys=None,
        dreams_dropout_prob=dreams_dropout_prob,
        use_length_condition=True,
        use_swiglu=True,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Create Lightning module
    print("[4/5] Creating Lightning module...")
    pl_module = MolecularFlowMatching(
        model=model,
        processor=processor,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{experiment_name}",
        filename='{epoch}-{step}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=save_top_k,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Logger
    logger = WandbLogger(
        project=project_name,
        name=experiment_name,
        log_model=True,
        save_dir="logs",
        id=wandb_id if resume_from_checkpoint else None,
        resume="allow" if resume_from_checkpoint else None,
    )
    
    # Trainer
    print("[5/5] Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps if max_epochs is None else -1,
        devices=devices,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        deterministic=False,
    )
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    # Train
    trainer.fit(pl_module, datamodule, ckpt_path=resume_from_checkpoint)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--data_dir", type=str, default="../data/massspecgym_filtered")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--dreams_dropout_prob", type=float, default=0.1)
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    
    # System
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    
    # Logging
    parser.add_argument("--project_name", type=str, default="molecular-generation")
    parser.add_argument("--experiment_name", type=str, default="molecular-dit")
    parser.add_argument("--wandb_id", type=str, default=None, help="WandB run ID to resume")
    
    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    train(**vars(args))
