"""
Loss functions for MFCT-GAN training
Includes LSGAN, Projection, Reconstruction, and Subjective losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSGANLoss(nn.Module):
    """Least Squares GAN Loss - alleviates vanishing gradient problem"""
    def __init__(self):
        super(LSGANLoss, self).__init__()

    def forward(self, discriminator_output, is_real=True):
        """
        Args:
            discriminator_output: Output from discriminator
            is_real: Boolean, True for real data, False for fake
        Returns:
            loss: LSGAN loss
        """
        if is_real:
            target = torch.ones_like(discriminator_output)
        else:
            target = torch.zeros_like(discriminator_output)

        return F.mse_loss(discriminator_output, target)


class ProjectionLoss(nn.Module):
    """Projection Loss - constrains geometric shape based on multiple projection planes"""
    def __init__(self):
        super(ProjectionLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, reconstructed_volume, target_volume):
        """
        Args:
            reconstructed_volume: Reconstructed 3D CT volume (B, 1, D, H, W)
            target_volume: Ground truth 3D CT volume (B, 1, D, H, W)
        Returns:
            loss: Sum of projection losses on three planes
        """
        # Project on different planes
        # Vertical plane (XY): average along Z
        proj_xy_recon = torch.mean(reconstructed_volume, dim=2, keepdim=True)
        proj_xy_target = torch.mean(target_volume, dim=2, keepdim=True)
        loss_xy = self.criterion(proj_xy_recon, proj_xy_target)

        # Horizontal plane (XZ): average along Y
        proj_xz_recon = torch.mean(reconstructed_volume, dim=3, keepdim=True)
        proj_xz_target = torch.mean(target_volume, dim=3, keepdim=True)
        loss_xz = self.criterion(proj_xz_recon, proj_xz_target)

        # Width plane (YZ): average along X
        proj_yz_recon = torch.mean(reconstructed_volume, dim=4, keepdim=True)
        proj_yz_target = torch.mean(target_volume, dim=4, keepdim=True)
        loss_yz = self.criterion(proj_yz_recon, proj_yz_target)

        total_loss = loss_xy + loss_xz + loss_yz
        return total_loss


class ReconstructionLoss(nn.Module):
    """Reconstruction Loss - pixel-level L1 loss to prevent blurring"""
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, reconstructed_volume, target_volume):
        """
        Args:
            reconstructed_volume: Reconstructed 3D CT volume
            target_volume: Ground truth 3D CT volume
        Returns:
            loss: L1 reconstruction loss
        """
        return self.criterion(reconstructed_volume, target_volume)


class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) Loss - improves visual quality"""
    def __init__(self, window_size=11, sigma=1.5, reduction='mean'):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.reduction = reduction

        # Create Gaussian kernel
        x = torch.arange(window_size).float() - (window_size - 1) / 2.0
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.unsqueeze(-1) * kernel_1d.unsqueeze(0)
        self.register_buffer('kernel_2d', kernel_2d.unsqueeze(0).unsqueeze(0))

    def forward(self, x, y):
        """
        Args:
            x: Reconstructed volume
            y: Target volume
        Returns:
            loss: 1 - SSIM (so minimizing the loss maximizes SSIM)
        """
        # Constants for numerical stability
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        # Pad input for padding mode
        pad = self.window_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad, pad, pad), mode='reflect')
        y_padded = F.pad(y, (pad, pad, pad, pad, pad, pad), mode='reflect')

        # Calculate means (simplified for 3D - using max projections)
        mu_x = torch.mean(x_padded, dim=(2, 3, 4), keepdim=True)
        mu_y = torch.mean(y_padded, dim=(2, 3, 4), keepdim=True)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        # Calculate variances
        sigma_x_sq = torch.var(x_padded, dim=(2, 3, 4), keepdim=True, unbiased=False)
        sigma_y_sq = torch.var(y_padded, dim=(2, 3, 4), keepdim=True, unbiased=False)
        sigma_xy = torch.mean((x_padded - mu_x) * (y_padded - mu_y), dim=(2, 3, 4), keepdim=True)

        # Calculate SSIM
        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim = numerator / denominator

        # Average SSIM
        if self.reduction == 'mean':
            return 1.0 - torch.mean(ssim)
        else:
            return 1.0 - ssim


class SubjectiveLoss(nn.Module):
    """Subjective Loss - combination of SSIM and Smooth L1 loss for visual quality"""
    def __init__(self, ssim_weight=0.9):
        super(SubjectiveLoss, self).__init__()
        self.ssim_weight = ssim_weight
        self.ssim_loss = SSIMLoss()
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean', beta=0.1)

    def forward(self, reconstructed_volume, target_volume):
        """
        Args:
            reconstructed_volume: Reconstructed 3D CT volume
            target_volume: Ground truth 3D CT volume
        Returns:
            loss: Weighted combination of SSIM and Smooth L1 loss
        """
        ssim_loss = self.ssim_loss(reconstructed_volume, target_volume)
        smooth_l1 = self.smooth_l1_loss(reconstructed_volume, target_volume)

        total_loss = self.ssim_weight * ssim_loss + (1 - self.ssim_weight) * smooth_l1
        return total_loss


class MFCT_GAN_Loss(nn.Module):
    """Total MFCT-GAN loss function combining all four loss components"""
    def __init__(self, alpha1=0.1, alpha2=8.0, alpha3=8.0, alpha4=2.0, ssim_weight=0.9):
        super(MFCT_GAN_Loss, self).__init__()
        self.alpha1 = alpha1  # LSGAN weight
        self.alpha2 = alpha2  # Projection weight
        self.alpha3 = alpha3  # Reconstruction weight
        self.alpha4 = alpha4  # Subjective weight

        self.lsgan_loss = LSGANLoss()
        self.projection_loss = ProjectionLoss()
        self.reconstruction_loss = ReconstructionLoss()
        self.subjective_loss = SubjectiveLoss(ssim_weight=ssim_weight)

    def generator_loss(self, discriminator_output, reconstructed_volume, target_volume):
        """
        Calculate total generator loss
        Args:
            discriminator_output: Output from discriminator for fake data
            reconstructed_volume: Generated 3D CT volume
            target_volume: Ground truth 3D CT volume
        Returns:
            loss_dict: Dictionary containing all loss components and total loss
        """
        # LSGAN adversarial loss (for fool the discriminator)
        loss_lsgan = self.lsgan_loss(discriminator_output, is_real=True)

        # Projection loss
        loss_projection = self.projection_loss(reconstructed_volume, target_volume)

        # Reconstruction loss
        loss_reconstruction = self.reconstruction_loss(reconstructed_volume, target_volume)

        # Subjective loss
        loss_subjective = self.subjective_loss(reconstructed_volume, target_volume)

        # Total generator loss
        total_loss = (
            self.alpha1 * loss_lsgan
            + self.alpha2 * loss_projection
            + self.alpha3 * loss_reconstruction
            + self.alpha4 * loss_subjective
        )

        loss_dict = {
            'total': total_loss,
            'lsgan': loss_lsgan,
            'projection': loss_projection,
            'reconstruction': loss_reconstruction,
            'subjective': loss_subjective,
        }

        return total_loss, loss_dict

    def discriminator_loss(self, discriminator_real, discriminator_fake):
        """
        Calculate discriminator loss
        Args:
            discriminator_real: Discriminator output for real data
            discriminator_fake: Discriminator output for fake data
        Returns:
            loss_dict: Dictionary containing discriminator loss components
        """
        loss_real = self.lsgan_loss(discriminator_real, is_real=True)
        loss_fake = self.lsgan_loss(discriminator_fake, is_real=False)

        total_loss = loss_real + loss_fake

        loss_dict = {
            'total': total_loss,
            'real': loss_real,
            'fake': loss_fake,
        }

        return total_loss, loss_dict
