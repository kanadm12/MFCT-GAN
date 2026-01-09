"""
MFCT-GAN Package
Multi-information Fusion Network for 3D CT Reconstruction
"""

from .models import (
    MFCT_GAN_Generator,
    PatchDiscriminator3D,
    DualParallelEncoder,
    Encoder2D,
    MultiChannelResidualDenseBlock,
    TransitionBlock,
    SkipConnectionModification,
    Decoder3D,
    Basic3DBlock,
)
from .losses import (
    MFCT_GAN_Loss,
    LSGANLoss,
    ProjectionLoss,
    ReconstructionLoss,
    SSIMLoss,
    SubjectiveLoss,
)
from .trainer import MFCT_GAN_Trainer
from .inference import MFCT_GAN_Inferencer, inference_single_sample
from .dataset import create_dataloaders, LIDCIDRI_Dataset, SyntheticDataset
from .config import Config, ModelConfig, TrainingConfig, DataConfig, ExperimentConfig

__version__ = '1.0.0'
__author__ = 'MFCT-GAN Implementation'

__all__ = [
    'MFCT_GAN_Generator',
    'PatchDiscriminator3D',
    'DualParallelEncoder',
    'Encoder2D',
    'MultiChannelResidualDenseBlock',
    'TransitionBlock',
    'SkipConnectionModification',
    'Decoder3D',
    'Basic3DBlock',
    'MFCT_GAN_Loss',
    'LSGANLoss',
    'ProjectionLoss',
    'ReconstructionLoss',
    'SSIMLoss',
    'SubjectiveLoss',
    'MFCT_GAN_Trainer',
    'MFCT_GAN_Inferencer',
    'inference_single_sample',
    'create_dataloaders',
    'LIDCIDRI_Dataset',
    'SyntheticDataset',
    'Config',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'ExperimentConfig',
]
