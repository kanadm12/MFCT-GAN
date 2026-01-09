"""
Main training script for MFCT-GAN
"""

import torch
import argparse
import os
from pathlib import Path

from models import MFCT_GAN_Generator, PatchDiscriminator3D
from losses import MFCT_GAN_Loss
from trainer import MFCT_GAN_Trainer
from dataset import create_dataloaders


def main(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create model
    print("Creating MFCT-GAN models...")
    generator = MFCT_GAN_Generator(base_channels=args.base_channels)
    discriminator = PatchDiscriminator3D(in_channels=1, base_channels=args.base_channels)
    
    # Print model info
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Create loss function
    loss_fn = MFCT_GAN_Loss(
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        alpha3=args.alpha3,
        alpha4=args.alpha4,
        ssim_weight=args.ssim_weight,
    )
    
    # Create trainer
    trainer = MFCT_GAN_Trainer(
        generator=generator,
        discriminator=discriminator,
        loss_fn=loss_fn,
        device=device,
        learning_rate_g=args.lr_g,
        learning_rate_d=args.lr_d,
        beta1=args.beta1,
        beta2=args.beta2,
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_synthetic=args.use_synthetic,
        use_drr_patient_data=args.use_drr_patient_data,
        num_synthetic_samples=args.num_synthetic_samples,
        x_ray_size=args.x_ray_size,
        ct_volume_size=args.ct_volume_size,
        train_val_split=args.train_val_split,
        vertical_flip=args.vertical_flip,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Train
    print("Starting training...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MFCT-GAN Training Script')
    
    # Model arguments
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels in the model')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    # Learning rate and optimizer arguments
    parser.add_argument('--lr_g', type=float, default=0.0002,
                        help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.0002,
                        help='Discriminator learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2')
    
    # Loss weights
    parser.add_argument('--alpha1', type=float, default=0.1,
                        help='LSGAN loss weight')
    parser.add_argument('--alpha2', type=float, default=8.0,
                        help='Projection loss weight')
    parser.add_argument('--alpha3', type=float, default=8.0,
                        help='Reconstruction loss weight')
    parser.add_argument('--alpha4', type=float, default=2.0,
                        help='Subjective loss weight')
    parser.add_argument('--ssim_weight', type=float, default=0.9,
                        help='SSIM weight within subjective loss')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to dataset directory')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic dataset for testing')
    parser.add_argument('--use_drr_patient_data', action='store_true',
                        help='Use DRR patient data format (runpod structure)')
    parser.add_argument('--num_synthetic_samples', type=int, default=100,
                        help='Number of synthetic samples')
    parser.add_argument('--x_ray_size', type=int, default=128,
                        help='Size of X-ray images')
    parser.add_argument('--ct_volume_size', type=int, default=128,
                        help='Size of CT volume')
    parser.add_argument('--vertical_flip', action='store_true', default=True,
                        help='Vertically flip DRR images during training')
    parser.add_argument('--train_val_split', type=float, default=0.8,
                        help='Train/validation split ratio')
    
    # Checkpoint and logging
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to load')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./runs',
                        help='Directory for TensorBoard logs')
    
    args = parser.parse_args()
    
    main(args)
