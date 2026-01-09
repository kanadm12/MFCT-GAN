"""
Utility functions and testing for MFCT-GAN
"""

import torch
import numpy as np
from models import MFCT_GAN_Generator, PatchDiscriminator3D


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(generator, discriminator):
    """Print summary of model architectures and parameter counts"""
    print("=" * 60)
    print("MFCT-GAN Model Summary")
    print("=" * 60)
    
    gen_params = count_parameters(generator)
    disc_params = count_parameters(discriminator)
    total_params = gen_params + disc_params
    
    print(f"\nGenerator Parameters: {gen_params:,}")
    print(f"Discriminator Parameters: {disc_params:,}")
    print(f"Total Parameters: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("Generator Architecture")
    print("=" * 60)
    print(generator)
    
    print("\n" + "=" * 60)
    print("Discriminator Architecture")
    print("=" * 60)
    print(discriminator)


def test_model_forward_pass(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    batch_size=2,
    x_ray_size=128,
    ct_volume_size=128,
):
    """Test forward pass of the model"""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    # Create models
    generator = MFCT_GAN_Generator(base_channels=32).to(device)
    discriminator = PatchDiscriminator3D(in_channels=1, base_channels=64).to(device)
    
    # Create dummy input
    x_ray1 = torch.randn(batch_size, 1, x_ray_size, x_ray_size).to(device)
    x_ray2 = torch.randn(batch_size, 1, x_ray_size, x_ray_size).to(device)
    ct_volume_real = torch.randn(batch_size, 1, ct_volume_size, ct_volume_size, ct_volume_size).to(device)
    
    print(f"\nInput shapes:")
    print(f"  X-ray 1: {x_ray1.shape}")
    print(f"  X-ray 2: {x_ray2.shape}")
    print(f"  CT Volume (real): {ct_volume_real.shape}")
    
    # Test generator
    with torch.no_grad():
        ct_volume_fake = generator(x_ray1, x_ray2)
    
    print(f"\nGenerator output shape: {ct_volume_fake.shape}")
    assert ct_volume_fake.shape == (batch_size, 1, ct_volume_size, ct_volume_size, ct_volume_size), \
        f"Expected shape {(batch_size, 1, ct_volume_size, ct_volume_size, ct_volume_size)}, got {ct_volume_fake.shape}"
    
    # Test discriminator
    with torch.no_grad():
        disc_real = discriminator(ct_volume_real)
        disc_fake = discriminator(ct_volume_fake)
    
    print(f"Discriminator real output shape: {disc_real.shape}")
    print(f"Discriminator fake output shape: {disc_fake.shape}")
    
    print("\n✓ Forward pass test passed!")
    
    return generator, discriminator


def test_losses(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Test loss functions"""
    from losses import MFCT_GAN_Loss
    
    print("\n" + "=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)
    
    batch_size = 2
    ct_volume_size = 128
    
    # Create loss function
    loss_fn = MFCT_GAN_Loss(
        alpha1=0.1,
        alpha2=8.0,
        alpha3=8.0,
        alpha4=2.0,
        ssim_weight=0.9,
    )
    
    # Create dummy data
    ct_volume_real = torch.randn(batch_size, 1, ct_volume_size, ct_volume_size, ct_volume_size).to(device)
    ct_volume_fake = torch.randn(batch_size, 1, ct_volume_size, ct_volume_size, ct_volume_size).to(device)
    disc_real = torch.randn(batch_size, 1, 8, 8, 8).to(device)
    disc_fake = torch.randn(batch_size, 1, 8, 8, 8).to(device)
    
    # Test generator loss
    g_loss, g_loss_dict = loss_fn.generator_loss(disc_fake, ct_volume_fake, ct_volume_real)
    
    print(f"\nGenerator Loss Components:")
    for key, value in g_loss_dict.items():
        print(f"  {key}: {value.item():.6f}")
    
    # Test discriminator loss
    d_loss, d_loss_dict = loss_fn.discriminator_loss(disc_real, disc_fake)
    
    print(f"\nDiscriminator Loss Components:")
    for key, value in d_loss_dict.items():
        print(f"  {key}: {value.item():.6f}")
    
    print("\n✓ Loss function test passed!")


def test_training_step(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Test a single training step"""
    from losses import MFCT_GAN_Loss
    from trainer import MFCT_GAN_Trainer
    
    print("\n" + "=" * 60)
    print("Testing Training Step")
    print("=" * 60)
    
    # Create models
    generator = MFCT_GAN_Generator(base_channels=32).to(device)
    discriminator = PatchDiscriminator3D(in_channels=1, base_channels=64).to(device)
    loss_fn = MFCT_GAN_Loss()
    
    # Create trainer
    trainer = MFCT_GAN_Trainer(
        generator=generator,
        discriminator=discriminator,
        loss_fn=loss_fn,
        device=device,
    )
    
    # Create dummy data
    batch_size = 2
    x_ray1 = torch.randn(batch_size, 1, 128, 128).to(device)
    x_ray2 = torch.randn(batch_size, 1, 128, 128).to(device)
    ct_volume = torch.randn(batch_size, 1, 128, 128, 128).to(device)
    
    # Test training step
    loss_dict = trainer.train_step(x_ray1, x_ray2, ct_volume)
    
    print(f"\nTraining step losses:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n✓ Training step test passed!")


def run_all_tests(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Run all tests"""
    print(f"\nRunning tests on device: {device}\n")
    
    try:
        # Test forward pass
        generator, discriminator = test_model_forward_pass(device)
        print_model_summary(generator, discriminator)
        
        # Test losses
        test_losses(device)
        
        # Test training step
        test_training_step(device)
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_all_tests(device)
