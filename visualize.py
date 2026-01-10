"""
Visualization script for MFCT-GAN
Generates and visualizes 3D CT reconstruction from DRR images for a random patient
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random
from pathlib import Path
import nibabel as nib

from models import MFCT_GAN_Generator
from dataset import DRR_PatientDataset


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, max_val=1.0):
    """Calculate SSIM between two images"""
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    
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


def visualize_reconstruction(pa_drr, lat_drr, ct_gt, ct_pred, save_path=None):
    """
    Visualize input DRRs, ground truth CT, and predicted CT
    
    Args:
        pa_drr: PA DRR image (1, H, W)
        lat_drr: Lateral DRR image (1, H, W)
        ct_gt: Ground truth CT volume (1, D, H, W)
        ct_pred: Predicted CT volume (1, D, H, W)
        save_path: Path to save the visualization
    """
    # Convert to numpy and squeeze channel dimension
    pa_drr = pa_drr.squeeze().cpu().numpy()
    lat_drr = lat_drr.squeeze().cpu().numpy()
    ct_gt = ct_gt.squeeze().cpu().numpy()
    ct_pred = ct_pred.squeeze().cpu().numpy()
    
    # Calculate metrics
    psnr = calculate_psnr(torch.tensor(ct_pred), torch.tensor(ct_gt))
    ssim = calculate_ssim(torch.tensor(ct_pred), torch.tensor(ct_gt))
    
    # Select middle slices from each axis
    depth, height, width = ct_gt.shape
    mid_d = depth // 2
    mid_h = height // 2
    mid_w = width // 2
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Title with metrics
    fig.suptitle(f'MFCT-GAN 3D CT Reconstruction | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}', 
                 fontsize=16, fontweight='bold')
    
    # Input DRRs
    ax1 = plt.subplot(3, 5, 1)
    ax1.imshow(pa_drr, cmap='gray')
    ax1.set_title('Input: PA DRR', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 5, 2)
    ax2.imshow(lat_drr, cmap='gray')
    ax2.set_title('Input: Lateral DRR', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Leave space
    ax3 = plt.subplot(3, 5, 3)
    ax3.axis('off')
    
    # Ground Truth CT - Axial slice
    ax4 = plt.subplot(3, 5, 6)
    ax4.imshow(ct_gt[mid_d, :, :], cmap='gray')
    ax4.set_title('Ground Truth\nAxial Slice', fontsize=11, fontweight='bold')
    ax4.axis('off')
    
    # Ground Truth CT - Sagittal slice
    ax5 = plt.subplot(3, 5, 7)
    ax5.imshow(ct_gt[:, :, mid_w], cmap='gray')
    ax5.set_title('Ground Truth\nSagittal Slice', fontsize=11, fontweight='bold')
    ax5.axis('off')
    
    # Ground Truth CT - Coronal slice
    ax6 = plt.subplot(3, 5, 8)
    ax6.imshow(ct_gt[:, mid_h, :], cmap='gray')
    ax6.set_title('Ground Truth\nCoronal Slice', fontsize=11, fontweight='bold')
    ax6.axis('off')
    
    # Predicted CT - Axial slice
    ax7 = plt.subplot(3, 5, 11)
    ax7.imshow(ct_pred[mid_d, :, :], cmap='gray')
    ax7.set_title('Predicted\nAxial Slice', fontsize=11, fontweight='bold')
    ax7.axis('off')
    
    # Predicted CT - Sagittal slice
    ax8 = plt.subplot(3, 5, 12)
    ax8.imshow(ct_pred[:, :, mid_w], cmap='gray')
    ax8.set_title('Predicted\nSagittal Slice', fontsize=11, fontweight='bold')
    ax8.axis('off')
    
    # Predicted CT - Coronal slice
    ax9 = plt.subplot(3, 5, 13)
    ax9.imshow(ct_pred[:, mid_h, :], cmap='gray')
    ax9.set_title('Predicted\nCoronal Slice', fontsize=11, fontweight='bold')
    ax9.axis('off')
    
    # Difference maps
    diff_axial = np.abs(ct_gt[mid_d, :, :] - ct_pred[mid_d, :, :])
    diff_sagittal = np.abs(ct_gt[:, :, mid_w] - ct_pred[:, :, mid_w])
    diff_coronal = np.abs(ct_gt[:, mid_h, :] - ct_pred[:, mid_h, :])
    
    ax10 = plt.subplot(3, 5, 14)
    im1 = ax10.imshow(diff_axial, cmap='hot')
    ax10.set_title('Difference\nAxial', fontsize=11, fontweight='bold')
    ax10.axis('off')
    plt.colorbar(im1, ax=ax10, fraction=0.046, pad=0.04)
    
    ax11 = plt.subplot(3, 5, 15)
    im2 = ax11.imshow(diff_sagittal, cmap='hot')
    ax11.set_title('Difference\nSagittal', fontsize=11, fontweight='bold')
    ax11.axis('off')
    plt.colorbar(im2, ax=ax11, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main(args):
    """Main inference and visualization function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    generator = MFCT_GAN_Generator(
        base_channels=args.base_channels,
        ct_volume_size=args.ct_volume_size
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # Load dataset
    print(f"Loading dataset from: {args.data_dir}")
    dataset = DRR_PatientDataset(
        data_dir=args.data_dir,
        x_ray_size=args.x_ray_size,
        ct_volume_size=args.ct_volume_size,
        split='val',  # Use validation split
        train_val_split=args.train_val_split,
        vertical_flip=args.vertical_flip,
        max_patients=args.max_patients,
    )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Select random sample or specific index
    if args.sample_idx is not None:
        idx = args.sample_idx
        print(f"Using sample index: {idx}")
    else:
        idx = random.randint(0, len(dataset) - 1)
        print(f"Randomly selected sample index: {idx}")
    
    # Get sample
    pa_drr, lat_drr, ct_gt = dataset[idx]
    
    # Add batch dimension and move to device
    pa_drr = pa_drr.unsqueeze(0).to(device)
    lat_drr = lat_drr.unsqueeze(0).to(device)
    ct_gt = ct_gt.unsqueeze(0).to(device)
    
    # Generate prediction
    print("Generating prediction...")
    with torch.no_grad():
        ct_pred = generator(pa_drr, lat_drr)
    
    print("Prediction complete!")
    
    # Calculate metrics
    psnr = calculate_psnr(ct_pred, ct_gt)
    ssim = calculate_ssim(ct_pred, ct_gt)
    
    print(f"\nMetrics:")
    print(f"  PSNR: {psnr:.4f} dB")
    print(f"  SSIM: {ssim:.4f}")
    
    # Visualize
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    save_path = output_dir / f'reconstruction_sample_{idx}.png'
    
    visualize_reconstruction(
        pa_drr[0], lat_drr[0], ct_gt[0], ct_pred[0],
        save_path=str(save_path)
    )
    
    # Save volumes as .npy and .nii.gz
    if args.save_volumes:
        ct_gt_np = ct_gt[0].cpu().numpy()
        ct_pred_np = ct_pred[0].cpu().numpy()
        
        # Save as numpy arrays
        np.save(output_dir / f'ct_gt_sample_{idx}.npy', ct_gt_np)
        np.save(output_dir / f'ct_pred_sample_{idx}.npy', ct_pred_np)
        
        # Save as NIfTI files for medical imaging software (3D Slicer, ITK-SNAP, etc.)
        # Remove channel dimension and convert to proper orientation
        ct_gt_nifti = nib.Nifti1Image(ct_gt_np.squeeze(), affine=np.eye(4))
        ct_pred_nifti = nib.Nifti1Image(ct_pred_np.squeeze(), affine=np.eye(4))
        
        nib.save(ct_gt_nifti, str(output_dir / f'ct_gt_sample_{idx}.nii.gz'))
        nib.save(ct_pred_nifti, str(output_dir / f'ct_pred_sample_{idx}.nii.gz'))
        
        print(f"\nVolumes saved to {output_dir}")
        print(f"  - Ground truth: ct_gt_sample_{idx}.nii.gz")
        print(f"  - Predicted: ct_pred_sample_{idx}.nii.gz")
        print(f"  - View in 3D Slicer, ITK-SNAP, or other NIfTI viewers")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MFCT-GAN Visualization Script')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_psnr_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--base_channels', type=int, default=16,
                        help='Base number of channels in the model')
    parser.add_argument('--ct_volume_size', type=int, default=64,
                        help='Size of CT volume')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./drr_patient_data',
                        help='Path to dataset directory')
    parser.add_argument('--x_ray_size', type=int, default=128,
                        help='Size of X-ray images')
    parser.add_argument('--vertical_flip', action='store_true', default=True,
                        help='Vertically flip DRR images')
    parser.add_argument('--train_val_split', type=float, default=0.8,
                        help='Train/validation split ratio')
    parser.add_argument('--max_patients', type=int, default=100,
                        help='Maximum number of patients to use')
    
    # Sample selection
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='Specific sample index to visualize (None for random)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--save_volumes', action='store_true',
                        help='Save predicted and GT volumes as .npy files')
    
    args = parser.parse_args()
    
    main(args)
