"""
Configuration and utility functions for MFCT-GAN
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    base_channels: int = 32
    x_ray_size: int = 128
    ct_volume_size: int = 128


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 100
    batch_size: int = 4
    num_workers: int = 0
    
    # Learning rates
    lr_g: float = 0.0002
    lr_d: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    
    # Loss weights (alpha parameters)
    alpha1: float = 0.1      # LSGAN
    alpha2: float = 8.0      # Projection
    alpha3: float = 8.0      # Reconstruction
    alpha4: float = 2.0      # Subjective
    ssim_weight: float = 0.9 # SSIM in subjective loss


@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: str = './data'
    use_synthetic: bool = True
    num_synthetic_samples: int = 100
    train_val_split: float = 0.8  # 80% train, 20% val


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './runs'
    save_frequency: int = 10  # Save checkpoint every N epochs
    resume_checkpoint: Optional[str] = None


class Config:
    """Master configuration class"""
    def __init__(
        self,
        model: Optional[ModelConfig] = None,
        training: Optional[TrainingConfig] = None,
        data: Optional[DataConfig] = None,
        experiment: Optional[ExperimentConfig] = None,
    ):
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.data = data or DataConfig()
        self.experiment = experiment or ExperimentConfig()
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'experiment': self.experiment.__dict__,
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {})),
        )


# Default configurations for different scenarios

def get_default_config():
    """Get default configuration"""
    return Config()


def get_debug_config():
    """Get configuration for debugging (small model, fewer epochs)"""
    return Config(
        model=ModelConfig(base_channels=16),
        training=TrainingConfig(
            num_epochs=5,
            batch_size=2,
        ),
        data=DataConfig(
            use_synthetic=True,
            num_synthetic_samples=50,
        ),
    )


def get_production_config():
    """Get configuration for production (larger model, more training)"""
    return Config(
        model=ModelConfig(base_channels=64),
        training=TrainingConfig(
            num_epochs=200,
            batch_size=8,
            lr_g=0.0001,
            lr_d=0.0001,
        ),
        data=DataConfig(
            use_synthetic=False,
        ),
        experiment=ExperimentConfig(
            save_frequency=5,
        ),
    )
