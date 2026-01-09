# MFCT-GAN Quick Start Guide

## Installation (5 minutes)

### Step 1: Clone/Extract the Project
```bash
cd /path/to/mfct_gan
```

### Step 2: Create Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Basic Usage (10 minutes)

### Option A: Test with Synthetic Data

```python
from mfct_gan import MFCT_GAN_Generator, PatchDiscriminator3D, MFCT_GAN_Loss, MFCT_GAN_Trainer
from mfct_gan import create_dataloaders

# Create models
generator = MFCT_GAN_Generator(base_channels=32)
discriminator = PatchDiscriminator3D(base_channels=64)

# Create loss and trainer
loss_fn = MFCT_GAN_Loss()
trainer = MFCT_GAN_Trainer(generator, discriminator, loss_fn)

# Create dataloaders (synthetic)
train_loader, val_loader = create_dataloaders(
    use_synthetic=True,
    num_synthetic_samples=100,
    batch_size=4
)

# Train
trainer.fit(train_loader, val_loader, num_epochs=10)
```

**Run from command line:**
```bash
python train.py --use_synthetic --batch_size 4 --num_epochs 10
```

### Option B: Run Examples

```bash
# All examples
python examples.py

# Specific example
python examples.py --example 1

# Unit tests
python examples.py --test
```

### Option C: Quick Test

```python
from mfct_gan.utils import run_all_tests
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
run_all_tests(device)
```

## Training with Real Data

### Step 1: Prepare Your Dataset

Create a directory structure:
```
data/
├── train/
│   ├── sample_0/
│   │   ├── xray1.npy      # (128, 128)
│   │   ├── xray2.npy      # (128, 128)
│   │   └── ct_volume.npy  # (128, 128, 128)
│   └── sample_1/
│       ├── xray1.npy
│       ├── xray2.npy
│       └── ct_volume.npy
├── val/
│   └── sample_0/
│       ├── xray1.npy
│       ├── xray2.npy
│       └── ct_volume.npy
└── test/
    └── sample_0/
        ├── xray1.npy
        ├── xray2.npy
        └── ct_volume.npy
```

**Note:** X-ray images and CT volumes must be normalized to [0, 1]

### Step 2: Train

```bash
python train.py \
    --data_dir ./data \
    --batch_size 8 \
    --num_epochs 100 \
    --checkpoint_dir ./checkpoints
```

### Step 3: Monitor Training

```bash
tensorboard --logdir ./runs
```

Then open browser to `http://localhost:6006`

## Inference

### Method 1: Using Inferencer Class

```python
from mfct_gan import MFCT_GAN_Inferencer
import numpy as np

# Load trained model
inferencer = MFCT_GAN_Inferencer('checkpoints/checkpoint_epoch_100.pt')

# Load X-ray images
x_ray1 = np.load('path/to/xray1.npy')
x_ray2 = np.load('path/to/xray2.npy')

# Predict CT volume
ct_volume = inferencer.predict_from_arrays(x_ray1, x_ray2)

# Save result
inferencer.save_prediction(ct_volume, 'output/ct_volume.npy')
```

### Method 2: Direct Inference

```python
from mfct_gan import MFCT_GAN_Inferencer

ct_volume = MFCT_GAN_Inferencer.inference_single_sample(
    checkpoint_path='checkpoints/model.pt',
    x_ray1_path='path/to/xray1.npy',
    x_ray2_path='path/to/xray2.npy',
    output_path='output/ct_volume.npy'
)
```

## Configuration

### Using Configuration Files

```python
from mfct_gan.config import get_debug_config, get_production_config, Config

# Debug configuration (fast training, small model)
config = get_debug_config()

# Production configuration (large model, long training)
config = get_production_config()

# Custom configuration
from mfct_gan.config import ModelConfig, TrainingConfig

config = Config(
    model=ModelConfig(base_channels=64),
    training=TrainingConfig(
        num_epochs=200,
        batch_size=8,
        lr_g=0.0001,
        lr_d=0.0001
    )
)
```

## Troubleshooting

### Problem: "CUDA out of memory"
```python
# Reduce batch size
train_loader, val_loader = create_dataloaders(batch_size=2)

# Or reduce model size
generator = MFCT_GAN_Generator(base_channels=16)
```

### Problem: "No module named 'mfct_gan'"
```bash
# Make sure you're in the correct directory
cd /path/to/mfct_gan

# Add to Python path in script
import sys
sys.path.insert(0, '.')
```

### Problem: "Training loss increases"
```python
# Reduce learning rates
trainer = MFCT_GAN_Trainer(
    ...,
    learning_rate_g=0.00005,
    learning_rate_d=0.00005
)
```

## Project Structure

```
mfct_gan/
├── __init__.py           # Package initialization
├── models.py             # Network architectures
├── losses.py             # Loss functions
├── trainer.py            # Training loop
├── inference.py          # Inference utilities
├── dataset.py            # Data loading
├── config.py             # Configuration
├── utils.py              # Testing & utilities
├── train.py              # Main training script
├── examples.py           # Example scripts
├── requirements.txt      # Dependencies
├── README.md             # Full documentation
└── API_REFERENCE.md      # API reference
```

## Key Parameters

### Model Architecture
- **base_channels**: 32-64 (higher = larger model, more memory)
- **input size**: 128×128 (X-rays)
- **output size**: 128×128×128 (CT volume)

### Training Hyperparameters
- **batch_size**: 4-16 (depends on GPU memory)
- **learning_rate_g**: 0.0002 (generator)
- **learning_rate_d**: 0.0002 (discriminator)
- **num_epochs**: 100-200

### Loss Weights
- **α₁** (LSGAN): 0.1
- **α₂** (Projection): 8.0
- **α₃** (Reconstruction): 8.0
- **α₄** (Subjective): 2.0
- **ω** (SSIM in subjective): 0.9

## Expected Results

### Synthetic Data (Quick Test)
- Training time: ~1 minute (on GPU)
- Loss convergence: 50-100 iterations

### Real Data (LIDC-IDRI, 920 samples)
- Training time: ~24 hours (RTX 2080, batch_size=8)
- PSNR: ~31 dB
- SSIM: ~0.676

## Next Steps

1. **Explore**: Run `examples.py` to understand the codebase
2. **Experiment**: Modify hyperparameters and observe effects
3. **Train**: Use your own dataset
4. **Evaluate**: Check metrics using standard medical imaging tools
5. **Deploy**: Use `MFCT_GAN_Inferencer` for production

## Additional Resources

- **Full Documentation**: See `README.md`
- **API Reference**: See `API_REFERENCE.md`
- **Code Examples**: Run `examples.py`
- **Unit Tests**: Run `python examples.py --test`

## Getting Help

### Check These Files First
1. README.md - Architecture and overview
2. API_REFERENCE.md - Function documentation
3. examples.py - Working code examples
4. utils.py - Testing and debugging

### Common Issues
- GPU memory: Reduce batch_size or base_channels
- Training divergence: Reduce learning rates
- Poor quality: Train longer or adjust loss weights

## Performance Tips

### Speed Up Training
- Increase batch_size (if GPU memory allows)
- Use num_workers in DataLoader
- Use mixed precision training (advanced)

### Reduce Memory Usage
- Decrease batch_size
- Reduce base_channels
- Use gradient checkpointing (advanced)

### Improve Results
- Increase training epochs
- Use data augmentation
- Adjust loss weight ratios
- Implement learning rate scheduling

---

**Happy Training! 🚀**

For detailed information, refer to the comprehensive documentation in README.md and API_REFERENCE.md
