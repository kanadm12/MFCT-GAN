"""
Training utilities and trainer class for MFCT-GAN
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, window_size=11, max_val=1.0):
    """Calculate Structural Similarity Index between two images"""
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    
    # Simple SSIM calculation (averaged over batch and spatial dimensions)
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.mean(img1 ** 2) - mu1_sq
    sigma2_sq = torch.mean(img2 ** 2) - mu2_sq
    sigma12 = torch.mean(img1 * img2) - mu1_mu2
    
    ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim.item()


class MFCT_GAN_Trainer:
    """Trainer class for MFCT-GAN"""
    def __init__(
        self,
        generator,
        discriminator,
        loss_fn,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate_g=0.0002,
        learning_rate_d=0.0002,
        beta1=0.5,
        beta2=0.999,
    ):
        """
        Args:
            generator: MFCT_GAN_Generator instance
            discriminator: PatchDiscriminator3D instance
            loss_fn: MFCT_GAN_Loss instance
            device: Computing device ('cuda' or 'cpu')
            learning_rate_g: Generator learning rate
            learning_rate_d: Discriminator learning rate
            beta1, beta2: Adam optimizer parameters
        """
        self.device = device
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.loss_fn = loss_fn
        
        # Optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=learning_rate_g,
            betas=(beta1, beta2)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate_d,
            betas=(beta1, beta2)
        )
        
        # Logging
        self.writer = None
        self.global_step = 0

    def init_tensorboard(self, log_dir='./runs'):
        """Initialize TensorBoard writer"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f'mfct_gan_{timestamp}'
        self.writer = SummaryWriter(os.path.join(log_dir, run_name))

    def train_step(self, x_ray1, x_ray2, ct_volume):
        """
        Single training step for both generator and discriminator
        
        Args:
            x_ray1: First X-ray image (B, 1, 128, 128)
            x_ray2: Second X-ray image (B, 1, 128, 128)
            ct_volume: Ground truth CT volume (B, 1, 128, 128, 128)
            
        Returns:
            loss_dict: Dictionary with generator and discriminator losses
        """
        batch_size = x_ray1.shape[0]
        
        x_ray1 = x_ray1.to(self.device)
        x_ray2 = x_ray2.to(self.device)
        ct_volume = ct_volume.to(self.device)
        
        # ============================================
        # Train Discriminator
        # ============================================
        self.discriminator.train()
        self.generator.eval()
        
        # Generate fake CT volume
        with torch.no_grad():
            fake_ct = self.generator(x_ray1, x_ray2)
        
        # Discriminator for real data
        discriminator_real = self.discriminator(ct_volume)
        
        # Discriminator for fake data
        discriminator_fake = self.discriminator(fake_ct.detach())
        
        # Calculate discriminator loss
        loss_d, loss_d_dict = self.loss_fn.discriminator_loss(
            discriminator_real, discriminator_fake
        )
        
        # Update discriminator
        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()
        
        # ============================================
        # Train Generator
        # ============================================
        self.generator.train()
        self.discriminator.eval()
        
        # Generate fake CT volume
        fake_ct = self.generator(x_ray1, x_ray2)
        
        # Discriminator output for fake data
        discriminator_fake = self.discriminator(fake_ct)
        
        # Calculate generator loss
        loss_g, loss_g_dict = self.loss_fn.generator_loss(
            discriminator_fake, fake_ct, ct_volume
        )
        
        # Update generator
        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()
        
        # Combine loss dictionaries
        loss_dict = {
            'g_total': loss_g_dict['total'].item(),
            'g_lsgan': loss_g_dict['lsgan'].item(),
            'g_projection': loss_g_dict['projection'].item(),
            'g_reconstruction': loss_g_dict['reconstruction'].item(),
            'g_subjective': loss_g_dict['subjective'].item(),
            'd_total': loss_d_dict['total'].item(),
            'd_real': loss_d_dict['real'].item(),
            'd_fake': loss_d_dict['fake'].item(),
        }
        
        return loss_dict

    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            avg_loss_dict: Dictionary with average losses for the epoch
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {key: 0 for key in [
            'g_total', 'g_lsgan', 'g_projection', 'g_reconstruction', 
            'g_subjective', 'd_total', 'd_real', 'd_fake'
        ]}
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (x_ray1, x_ray2, ct_volume) in enumerate(pbar):
            loss_dict = self.train_step(x_ray1, x_ray2, ct_volume)
            
            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value
            
            # Update progress bar
            pbar.set_postfix({
                'g_loss': loss_dict['g_total'],
                'd_loss': loss_dict['d_total']
            })
            
            # Log to tensorboard
            if self.writer is not None:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Loss/{key}', value, self.global_step)
                self.global_step += 1
        
        # Calculate average losses
        num_batches = len(train_loader)
        avg_loss_dict = {key: value / num_batches for key, value in epoch_losses.items()}
        
        return avg_loss_dict

    def validate(self, val_loader):
        """
        Validate generator performance
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            avg_loss_dict: Dictionary with average validation losses and metrics
        """
        self.generator.eval()
        self.discriminator.eval()
        
        val_losses = {key: 0 for key in [
            'g_total', 'g_lsgan', 'g_projection', 'g_reconstruction', 
            'g_subjective', 'd_total', 'd_real', 'd_fake'
        ]}
        
        total_psnr = 0
        total_ssim = 0
        num_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, (x_ray1, x_ray2, ct_volume) in enumerate(pbar):
                x_ray1 = x_ray1.to(self.device)
                x_ray2 = x_ray2.to(self.device)
                ct_volume = ct_volume.to(self.device)
                
                # Generate fake CT
                fake_ct = self.generator(x_ray1, x_ray2)
                
                # Calculate PSNR and SSIM
                for i in range(fake_ct.shape[0]):
                    psnr = calculate_psnr(fake_ct[i], ct_volume[i])
                    ssim = calculate_ssim(fake_ct[i], ct_volume[i])
                    total_psnr += psnr
                    total_ssim += ssim
                    num_samples += 1
                
                # Discriminator outputs
                discriminator_real = self.discriminator(ct_volume)
                discriminator_fake = self.discriminator(fake_ct)
                
                # Calculate losses
                loss_g, loss_g_dict = self.loss_fn.generator_loss(
                    discriminator_fake, fake_ct, ct_volume
                )
                loss_d, loss_d_dict = self.loss_fn.discriminator_loss(
                    discriminator_real, discriminator_fake
                )
                
                val_losses['g_total'] += loss_g_dict['total'].item()
                val_losses['g_lsgan'] += loss_g_dict['lsgan'].item()
                val_losses['g_projection'] += loss_g_dict['projection'].item()
                val_losses['g_reconstruction'] += loss_g_dict['reconstruction'].item()
                val_losses['g_subjective'] += loss_g_dict['subjective'].item()
                val_losses['d_total'] += loss_d_dict['total'].item()
                val_losses['d_real'] += loss_d_dict['real'].item()
                val_losses['d_fake'] += loss_d_dict['fake'].item()
        
        # Calculate average losses and metrics
        num_batches = len(val_loader)
        avg_loss_dict = {key: value / num_batches for key, value in val_losses.items()}
        avg_loss_dict['psnr'] = total_psnr / num_samples
        avg_loss_dict['ssim'] = total_ssim / num_samples
        
        return avg_loss_dict

    def save_checkpoint(self, save_path, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']

    def fit(self, train_loader, val_loader=None, num_epochs=100, checkpoint_dir='./checkpoints'):
        """
        Train the model for multiple epochs
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.init_tensorboard()
        
        for epoch in range(num_epochs):
            # Training
            train_losses = self.train_epoch(train_loader, epoch)
            print(f"\nEpoch {epoch} - Train Losses:")
            for key, value in train_losses.items():
                print(f"  {key}: {value:.6f}")
            
            # Validation
            if val_loader is not None:
                val_losses = self.validate(val_loader)
                print(f"Epoch {epoch} - Val Losses:")
                for key, value in val_losses.items():
                    print(f"  {key}: {value:.6f}")
                    if self.writer:
                        self.writer.add_scalar(f'Val_Loss/{key}', value, epoch)
                
                # Print metrics prominently
                print(f"\n{'='*50}")
                print(f"Epoch {epoch} Metrics:")
                print(f"  PSNR: {val_losses['psnr']:.4f} dB")
                print(f"  SSIM: {val_losses['ssim']:.4f}")
                print(f"{'='*50}\n")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
                self.save_checkpoint(checkpoint_path, epoch)
        
        if self.writer:
            self.writer.close()

    def predict(self, x_ray1, x_ray2):
        """
        Generate CT volume prediction
        
        Args:
            x_ray1: First X-ray image (B, 1, 128, 128)
            x_ray2: Second X-ray image (B, 1, 128, 128)
            
        Returns:
            ct_volume: Predicted 3D CT volume (B, 1, 128, 128, 128)
        """
        self.generator.eval()
        x_ray1 = x_ray1.to(self.device)
        x_ray2 = x_ray2.to(self.device)
        
        with torch.no_grad():
            ct_volume = self.generator(x_ray1, x_ray2)
        
        return ct_volume
