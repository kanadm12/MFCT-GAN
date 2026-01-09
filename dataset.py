"""
Data loading utilities for MFCT-GAN
Includes dataset classes and data augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import nibabel as nib
from PIL import Image


class LIDCIDRI_Dataset(Dataset):
    """
    Dataset class for LIDC-IDRI lung image database
    Expects paired 2D projections (biplanar X-rays) and 3D CT volumes
    """
    def __init__(
        self,
        data_dir,
        x_ray_size=128,
        ct_volume_size=128,
        split='train',
        transform=None,
    ):
        """
        Args:
            data_dir: Directory containing dataset files
            x_ray_size: Size of X-ray images (H, W) - default 128
            ct_volume_size: Size of CT volume (D, H, W) - default 128
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply to data
        """
        self.data_dir = Path(data_dir)
        self.x_ray_size = x_ray_size
        self.ct_volume_size = ct_volume_size
        self.split = split
        self.transform = transform
        
        # Load file lists
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load sample file paths"""
        samples = []
        split_dir = self.data_dir / self.split
        
        if split_dir.exists():
            # Assuming structure: split/sample_*/
            for sample_dir in sorted(split_dir.glob('sample_*')):
                sample_dict = {
                    'xray1': sample_dir / 'xray1.npy',
                    'xray2': sample_dir / 'xray2.npy',
                    'ct_volume': sample_dir / 'ct_volume.npy',
                }
                # Check if all files exist
                if all(f.exists() for f in sample_dict.values()):
                    samples.append(sample_dict)
        
        if not samples:
            print(f"Warning: No samples found in {split_dir}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Args:
            idx: Index of sample
            
        Returns:
            x_ray1: First X-ray image (1, H, W)
            x_ray2: Second X-ray image (1, H, W)
            ct_volume: CT volume (1, D, H, W)
        """
        sample_dict = self.samples[idx]
        
        # Load data
        x_ray1 = np.load(sample_dict['xray1']).astype(np.float32)
        x_ray2 = np.load(sample_dict['xray2']).astype(np.float32)
        ct_volume = np.load(sample_dict['ct_volume']).astype(np.float32)
        
        # Ensure correct shapes and sizes
        x_ray1 = self._resize_2d(x_ray1, self.x_ray_size)
        x_ray2 = self._resize_2d(x_ray2, self.x_ray_size)
        ct_volume = self._resize_3d(ct_volume, self.ct_volume_size)
        
        # Add channel dimension if needed
        if x_ray1.ndim == 2:
            x_ray1 = np.expand_dims(x_ray1, axis=0)
        if x_ray2.ndim == 2:
            x_ray2 = np.expand_dims(x_ray2, axis=0)
        if ct_volume.ndim == 3:
            ct_volume = np.expand_dims(ct_volume, axis=0)
        
        # Normalize to [0, 1]
        x_ray1 = np.clip(x_ray1, 0, 255) / 255.0
        x_ray2 = np.clip(x_ray2, 0, 255) / 255.0
        ct_volume = np.clip(ct_volume, 0, 255) / 255.0
        
        # Apply transforms if any
        if self.transform:
            x_ray1 = self.transform(x_ray1)
            x_ray2 = self.transform(x_ray2)
            ct_volume = self.transform(ct_volume)
        
        # Convert to tensors
        x_ray1 = torch.from_numpy(x_ray1)
        x_ray2 = torch.from_numpy(x_ray2)
        ct_volume = torch.from_numpy(ct_volume)
        
        return x_ray1, x_ray2, ct_volume
    
    @staticmethod
    def _resize_2d(img, target_size):
        """Resize 2D image to target size"""
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        
        if img.shape != target_size:
            from scipy import ndimage
            zoom_factors = (target_size[0] / img.shape[0], target_size[1] / img.shape[1])
            img = ndimage.zoom(img, zoom_factors, order=1)
        
        return img
    
    @staticmethod
    def _resize_3d(volume, target_size):
        """Resize 3D volume to target size"""
        if isinstance(target_size, int):
            target_size = (target_size, target_size, target_size)
        
        if volume.shape != target_size:
            from scipy import ndimage
            zoom_factors = tuple(target_size[i] / volume.shape[i] for i in range(3))
            volume = ndimage.zoom(volume, zoom_factors, order=1)
        
        return volume


class DRR_PatientDataset(Dataset):
    """
    Dataset class for DRR patient data from runpod
    Loads CT volumes (.nii.gz) and corresponding DRR images (.png)
    Applies vertical flip to DRR images during training
    
    Expected structure:
        data_dir/
        ├── patient_id_1/
        │   ├── patient_id_1.nii.gz
        │   ├── patient_id_1_pa_drr.png
        │   └── patient_id_1_lat_drr.png
        └── patient_id_2/
            └── ...
    """
    def __init__(
        self,
        data_dir,
        x_ray_size=128,
        ct_volume_size=128,
        split='train',
        train_val_split=0.8,
        vertical_flip=True,
        max_patients=None,
        transform=None,
    ):
        """
        Args:
            data_dir: Directory containing patient folders
            x_ray_size: Size of DRR images (H, W) - default 128
            ct_volume_size: Size of CT volume (D, H, W) - default 128
            split: 'train' or 'val'
            train_val_split: Train/validation split ratio (default: 0.8)
            vertical_flip: Whether to vertically flip DRR images (default: True)
            max_patients: Maximum number of patients to use (None = use all)
            transform: Optional transform to apply to data
        """
        self.data_dir = Path(data_dir)
        self.x_ray_size = x_ray_size
        self.ct_volume_size = ct_volume_size
        self.split = split
        self.train_val_split = train_val_split
        self.vertical_flip = vertical_flip
        self.max_patients = max_patients
        self.transform = transform
        
        # Load file lists
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        
    def _load_samples(self):
        """Load sample file paths from patient directories"""
        samples = []
        
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return samples
        
        # Get all patient directories
        patient_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        # Limit to max_patients if specified
        if self.max_patients is not None:
            patient_dirs = patient_dirs[:self.max_patients]
        
        # Split into train/val
        split_idx = int(len(patient_dirs) * self.train_val_split)
        if self.split == 'train':
            patient_dirs = patient_dirs[:split_idx]
        else:
            patient_dirs = patient_dirs[split_idx:]
        
        # Load samples from each patient directory
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            
            # Define expected file paths
            ct_path = patient_dir / f"{patient_id}.nii.gz"
            pa_drr_path = patient_dir / f"{patient_id}_pa_drr.png"
            lat_drr_path = patient_dir / f"{patient_id}_lat_drr.png"
            
            # Check if all files exist
            if ct_path.exists() and pa_drr_path.exists() and lat_drr_path.exists():
                samples.append({
                    'ct_volume': ct_path,
                    'pa_drr': pa_drr_path,
                    'lat_drr': lat_drr_path,
                })
            else:
                print(f"Warning: Missing files for patient {patient_id}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Args:
            idx: Index of sample
            
        Returns:
            pa_drr: PA (posterior-anterior) DRR image (1, H, W)
            lat_drr: Lateral DRR image (1, H, W)
            ct_volume: CT volume (1, D, H, W)
        """
        sample_dict = self.samples[idx]
        
        # Load CT volume from .nii.gz
        ct_nifti = nib.load(str(sample_dict['ct_volume']))
        ct_volume = ct_nifti.get_fdata().astype(np.float32)
        
        # Load DRR images from .png
        pa_drr = np.array(Image.open(sample_dict['pa_drr']).convert('L')).astype(np.float32)
        lat_drr = np.array(Image.open(sample_dict['lat_drr']).convert('L')).astype(np.float32)
        
        # Apply vertical flip if enabled
        if self.vertical_flip:
            pa_drr = np.flipud(pa_drr)
            lat_drr = np.flipud(lat_drr)
        
        # Resize images and volumes
        pa_drr = self._resize_2d(pa_drr, self.x_ray_size)
        lat_drr = self._resize_2d(lat_drr, self.x_ray_size)
        ct_volume = self._resize_3d(ct_volume, self.ct_volume_size)
        
        # Add channel dimension
        pa_drr = np.expand_dims(pa_drr, axis=0)
        lat_drr = np.expand_dims(lat_drr, axis=0)
        ct_volume = np.expand_dims(ct_volume, axis=0)
        
        # Normalize to [0, 1]
        pa_drr = (pa_drr - pa_drr.min()) / (pa_drr.max() - pa_drr.min() + 1e-8)
        lat_drr = (lat_drr - lat_drr.min()) / (lat_drr.max() - lat_drr.min() + 1e-8)
        ct_volume = (ct_volume - ct_volume.min()) / (ct_volume.max() - ct_volume.min() + 1e-8)
        
        # Apply transforms if any
        if self.transform:
            pa_drr = self.transform(pa_drr)
            lat_drr = self.transform(lat_drr)
            ct_volume = self.transform(ct_volume)
        
        # Convert to tensors
        pa_drr = torch.from_numpy(pa_drr)
        lat_drr = torch.from_numpy(lat_drr)
        ct_volume = torch.from_numpy(ct_volume)
        
        return pa_drr, lat_drr, ct_volume
    
    @staticmethod
    def _resize_2d(img, target_size):
        """Resize 2D image to target size"""
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        
        if img.shape != target_size:
            from scipy import ndimage
            zoom_factors = (target_size[0] / img.shape[0], target_size[1] / img.shape[1])
            img = ndimage.zoom(img, zoom_factors, order=1)
        
        return img
    
    @staticmethod
    def _resize_3d(volume, target_size):
        """Resize 3D volume to target size"""
        if isinstance(target_size, int):
            target_size = (target_size, target_size, target_size)
        
        if volume.shape != target_size:
            from scipy import ndimage
            zoom_factors = tuple(target_size[i] / volume.shape[i] for i in range(3))
            volume = ndimage.zoom(volume, zoom_factors, order=1)
        
        return volume


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing purposes
    Generates random biplanar X-rays and corresponding CT volumes
    """
    def __init__(self, num_samples=100, x_ray_size=128, ct_volume_size=128):
        """
        Args:
            num_samples: Number of synthetic samples to generate
            x_ray_size: Size of X-ray images
            ct_volume_size: Size of CT volume
        """
        self.num_samples = num_samples
        self.x_ray_size = x_ray_size
        self.ct_volume_size = ct_volume_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Generate synthetic data"""
        # Generate random X-ray images
        x_ray1 = torch.randn(1, self.x_ray_size, self.x_ray_size).abs()
        x_ray2 = torch.randn(1, self.x_ray_size, self.x_ray_size).abs()
        
        # Generate corresponding CT volume (simplified correlation)
        ct_volume = torch.randn(1, self.ct_volume_size, self.ct_volume_size, self.ct_volume_size).abs()
        
        # Normalize to [0, 1]
        x_ray1 = (x_ray1 - x_ray1.min()) / (x_ray1.max() - x_ray1.min() + 1e-8)
        x_ray2 = (x_ray2 - x_ray2.min()) / (x_ray2.max() - x_ray2.min() + 1e-8)
        ct_volume = (ct_volume - ct_volume.min()) / (ct_volume.max() - ct_volume.min() + 1e-8)
        
        return x_ray1, x_ray2, ct_volume


def create_dataloaders(
    data_dir=None,
    batch_size=4,
    num_workers=0,
    use_synthetic=False,
    use_drr_patient_data=False,
    num_synthetic_samples=100,
    x_ray_size=128,
    ct_volume_size=128,
    train_val_split=0.8,
    vertical_flip=True,
    max_patients=None,
):
    """
    Create dataloaders for training and validation
    
    Args:
        data_dir: Directory containing dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        use_synthetic: Whether to use synthetic dataset (for testing)
        use_drr_patient_data: Whether to use DRR patient data format
        num_synthetic_samples: Number of synthetic samples if use_synthetic=True
        x_ray_size: Size of X-ray images
        ct_volume_size: Size of CT volume
        train_val_split: Train/val split ratio (default: 0.8)
        vertical_flip: Whether to vertically flip DRRs (only for DRR patient data)
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    if use_synthetic:
        # Create synthetic datasets
        train_dataset = SyntheticDataset(
            num_samples=num_synthetic_samples,
            x_ray_size=x_ray_size,
            ct_volume_size=ct_volume_size
        )
        val_dataset = SyntheticDataset(
            num_samples=num_synthetic_samples // 5,
            x_ray_size=x_ray_size,
            ct_volume_size=ct_volume_size
        )
    elif use_drr_patient_data:
        # Load DRR patient data
        train_dataset = DRR_PatientDataset(
            data_dir=data_dir,
            x_ray_size=x_ray_size,
            ct_volume_size=ct_volume_size,
            split='train',
            train_val_split=train_val_split,
            vertical_flip=vertical_flip,
            max_patients=max_patients,
        )
        val_dataset = DRR_PatientDataset(
            data_dir=data_dir,
            x_ray_size=x_ray_size,
            ct_volume_size=ct_volume_size,
            split='val',
            train_val_split=train_val_split,
            vertical_flip=vertical_flip,
            max_patients=max_patients,
        )
    else:
        # Load real datasets (LIDC-IDRI format)
        train_dataset = LIDCIDRI_Dataset(
            data_dir=data_dir,
            x_ray_size=x_ray_size,
            ct_volume_size=ct_volume_size,
            split='train',
        )
        val_dataset = LIDCIDRI_Dataset(
            data_dir=data_dir,
            x_ray_size=x_ray_size,
            ct_volume_size=ct_volume_size,
            split='val',
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
    )
    
    return train_loader, val_loader
