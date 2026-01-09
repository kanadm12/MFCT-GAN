"""
Example scripts for MFCT-GAN usage
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mfct_gan import (
    MFCT_GAN_Generator,
    PatchDiscriminator3D,
    MFCT_GAN_Loss,
    MFCT_GAN_Trainer,
    create_dataloaders,
    MFCT_GAN_Inferencer,
)
from mfct_gan.utils import run_all_tests, print_model_summary


def example_1_quick_test():
    """Example 1: Quick test with synthetic data"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Quick Test with Synthetic Data")
    print("="*70)
    
    # Create models
    generator = MFCT_GAN_Generator(base_channels=32)
    discriminator = PatchDiscriminator3D(base_channels=64)
    
    # Print model info
    print_model_summary(generator, discriminator)
    
    # Create dummy batch
    batch_size = 2
    x_ray1 = torch.randn(batch_size, 1, 128, 128)
    x_ray2 = torch.randn(batch_size, 1, 128, 128)
    
    # Forward pass
    with torch.no_grad():
        ct_volume = generator(x_ray1, x_ray2)
    
    print(f"\nInput X-ray 1 shape: {x_ray1.shape}")
    print(f"Input X-ray 2 shape: {x_ray2.shape}")
    print(f"Output CT volume shape: {ct_volume.shape}")
    print("\n✓ Example 1 completed successfully!")


def example_2_training_loop():
    """Example 2: Training loop with synthetic data"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Training Loop with Synthetic Data")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create models
    generator = MFCT_GAN_Generator(base_channels=32)
    discriminator = PatchDiscriminator3D(base_channels=64)
    loss_fn = MFCT_GAN_Loss(alpha1=0.1, alpha2=8.0, alpha3=8.0, alpha4=2.0)
    
    # Create trainer
    trainer = MFCT_GAN_Trainer(
        generator=generator,
        discriminator=discriminator,
        loss_fn=loss_fn,
        device=device,
        learning_rate_g=0.0002,
        learning_rate_d=0.0002,
    )
    
    # Create synthetic dataloaders
    train_loader, val_loader = create_dataloaders(
        use_synthetic=True,
        num_synthetic_samples=20,  # Small for example
        batch_size=2,
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Train for a few iterations
    print("\nTraining 2 epochs...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
        checkpoint_dir='./example_checkpoints',
    )
    
    print("\n✓ Example 2 completed successfully!")


def example_3_inference():
    """Example 3: Inference with trained model"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Inference")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create a dummy trained model for demonstration
    generator = MFCT_GAN_Generator(base_channels=32)
    
    # Create dummy checkpoint (in real scenario, this would be a trained model)
    checkpoint = {
        'generator': generator.state_dict(),
        'epoch': 100,
    }
    checkpoint_path = './example_checkpoints/example_model.pt'
    Path('./example_checkpoints').mkdir(exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    
    # Load for inference
    inferencer = MFCT_GAN_Inferencer(checkpoint_path, device=device)
    
    # Create dummy X-ray images
    x_ray1 = np.random.randn(128, 128).astype(np.float32)
    x_ray2 = np.random.randn(128, 128).astype(np.float32)
    
    # Predict
    ct_volume = inferencer.predict_from_arrays(x_ray1, x_ray2)
    
    print(f"Input X-ray 1 shape: {x_ray1.shape}")
    print(f"Input X-ray 2 shape: {x_ray2.shape}")
    print(f"Output CT volume shape: {ct_volume.shape}")
    print(f"CT volume value range: [{ct_volume.min():.4f}, {ct_volume.max():.4f}]")
    
    # Save output
    output_path = './example_checkpoints/output_ct_volume.npy'
    inferencer.save_prediction(ct_volume, output_path)
    
    print("\n✓ Example 3 completed successfully!")


def example_4_loss_functions():
    """Example 4: Understanding loss functions"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Loss Functions")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create loss function with specified weights
    loss_fn = MFCT_GAN_Loss(
        alpha1=0.1,   # LSGAN
        alpha2=8.0,   # Projection
        alpha3=8.0,   # Reconstruction
        alpha4=2.0,   # Subjective
        ssim_weight=0.9,
    )
    
    print("\nLoss Function Configuration:")
    print(f"  LSGAN weight (α1): {loss_fn.alpha1}")
    print(f"  Projection weight (α2): {loss_fn.alpha2}")
    print(f"  Reconstruction weight (α3): {loss_fn.alpha3}")
    print(f"  Subjective weight (α4): {loss_fn.alpha4}")
    print(f"  SSIM weight (ω): {loss_fn.subjective_loss.ssim_weight}")
    
    # Create dummy predictions
    batch_size = 2
    ct_size = 64  # Smaller for faster computation
    
    reconstructed = torch.randn(batch_size, 1, ct_size, ct_size, ct_size).to(device)
    target = torch.randn(batch_size, 1, ct_size, ct_size, ct_size).to(device)
    disc_fake = torch.randn(batch_size, 1, 8, 8, 8).to(device)
    disc_real = torch.randn(batch_size, 1, 8, 8, 8).to(device)
    
    # Calculate losses
    g_loss, g_losses = loss_fn.generator_loss(disc_fake, reconstructed, target)
    d_loss, d_losses = loss_fn.discriminator_loss(disc_real, disc_fake)
    
    print("\nGenerator Loss Components:")
    for name, value in g_losses.items():
        print(f"  {name}: {value.item():.6f}")
    
    print("\nDiscriminator Loss Components:")
    for name, value in d_losses.items():
        print(f"  {name}: {value.item():.6f}")
    
    print("\n✓ Example 4 completed successfully!")


def example_5_model_configuration():
    """Example 5: Different model configurations"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Model Configurations")
    print("="*70)
    
    configs = [
        ('Lightweight', 16),
        ('Standard', 32),
        ('Large', 64),
    ]
    
    for config_name, base_channels in configs:
        generator = MFCT_GAN_Generator(base_channels=base_channels)
        discriminator = PatchDiscriminator3D(base_channels=base_channels)
        
        gen_params = sum(p.numel() for p in generator.parameters())
        disc_params = sum(p.numel() for p in discriminator.parameters())
        total_params = gen_params + disc_params
        
        print(f"\n{config_name} Configuration (base_channels={base_channels}):")
        print(f"  Generator parameters: {gen_params:,}")
        print(f"  Discriminator parameters: {disc_params:,}")
        print(f"  Total parameters: {total_params:,}")
    
    print("\n✓ Example 5 completed successfully!")


def run_all_examples():
    """Run all examples"""
    try:
        example_1_quick_test()
        example_4_loss_functions()
        example_5_model_configuration()
        
        # These examples require more computation, run with caution
        # example_2_training_loop()
        # example_3_inference()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY ✓")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MFCT-GAN Examples')
    parser.add_argument('--example', type=int, default=None,
                        help='Run specific example (1-5), or all if not specified')
    parser.add_argument('--test', action='store_true',
                        help='Run unit tests instead of examples')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running unit tests...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        run_all_tests(device)
    elif args.example:
        example_func = {
            1: example_1_quick_test,
            2: example_2_training_loop,
            3: example_3_inference,
            4: example_4_loss_functions,
            5: example_5_model_configuration,
        }.get(args.example)
        
        if example_func:
            example_func()
        else:
            print(f"Example {args.example} not found")
    else:
        run_all_examples()
