"""
Inference utilities for MFCT-GAN
"""

import torch
import numpy as np
from pathlib import Path
from models import MFCT_GAN_Generator


class MFCT_GAN_Inferencer:
    """Inference class for MFCT-GAN"""
    
    def __init__(self, checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Computing device
        """
        self.device = torch.device(device)
        self.generator = MFCT_GAN_Generator(base_channels=32).to(self.device)
        self.load_checkpoint(checkpoint_path)
        self.generator.eval()
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'generator' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator'])
        else:
            self.generator.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def predict_from_arrays(self, x_ray1, x_ray2):
        """
        Predict CT volume from numpy arrays
        
        Args:
            x_ray1: First X-ray image as numpy array (H, W) or (1, H, W)
            x_ray2: Second X-ray image as numpy array (H, W) or (1, H, W)
            
        Returns:
            ct_volume: Predicted CT volume as numpy array (1, 128, 128, 128)
        """
        # Convert to tensors
        if isinstance(x_ray1, np.ndarray):
            x_ray1 = torch.from_numpy(x_ray1).float()
        if isinstance(x_ray2, np.ndarray):
            x_ray2 = torch.from_numpy(x_ray2).float()
        
        # Add batch dimension if needed
        if x_ray1.ndim == 2:
            x_ray1 = x_ray1.unsqueeze(0)
        if x_ray2.ndim == 2:
            x_ray2 = x_ray2.unsqueeze(0)
        if x_ray1.ndim == 3:
            x_ray1 = x_ray1.unsqueeze(0)
        if x_ray2.ndim == 3:
            x_ray2 = x_ray2.unsqueeze(0)
        
        # Move to device and predict
        x_ray1 = x_ray1.to(self.device)
        x_ray2 = x_ray2.to(self.device)
        
        with torch.no_grad():
            ct_volume = self.generator(x_ray1, x_ray2)
        
        # Convert back to numpy
        ct_volume = ct_volume.cpu().numpy()
        
        return ct_volume
    
    def predict_from_files(self, x_ray1_path, x_ray2_path):
        """
        Predict CT volume from image files
        
        Args:
            x_ray1_path: Path to first X-ray image
            x_ray2_path: Path to second X-ray image
            
        Returns:
            ct_volume: Predicted CT volume
        """
        # Load images
        x_ray1 = np.load(x_ray1_path).astype(np.float32)
        x_ray2 = np.load(x_ray2_path).astype(np.float32)
        
        return self.predict_from_arrays(x_ray1, x_ray2)
    
    def save_prediction(self, ct_volume, output_path):
        """
        Save predicted CT volume
        
        Args:
            ct_volume: CT volume to save
            output_path: Path to save the volume
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(ct_volume, torch.Tensor):
            ct_volume = ct_volume.cpu().numpy()
        
        np.save(output_path, ct_volume)
        print(f"Saved CT volume to {output_path}")


def inference_single_sample(
    checkpoint_path,
    x_ray1_path,
    x_ray2_path,
    output_path,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Inference on a single sample
    
    Args:
        checkpoint_path: Path to trained model
        x_ray1_path: Path to first X-ray image
        x_ray2_path: Path to second X-ray image
        output_path: Path to save output
        device: Computing device
    """
    inferencer = MFCT_GAN_Inferencer(checkpoint_path, device)
    ct_volume = inferencer.predict_from_files(x_ray1_path, x_ray2_path)
    inferencer.save_prediction(ct_volume, output_path)
    return ct_volume
