"""
MFCT-GAN Getting Started Script
Run this script to verify installation and test the system
"""

import sys
import platform
import subprocess


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_success(text):
    """Print success message"""
    print(f"✓ {text}")


def print_error(text):
    """Print error message"""
    print(f"✗ {text}")


def print_info(text):
    """Print info message"""
    print(f"  {text}")


def check_python():
    """Check Python version"""
    print_header("Python Environment")
    version = sys.version_info
    print_info(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    print_info(f"Platform: {platform.platform()}")
    
    if version.major >= 3 and version.minor >= 8:
        print_success("Python version is compatible")
        return True
    else:
        print_error("Python 3.8 or higher is required")
        return False


def check_packages():
    """Check if required packages are installed"""
    print_header("Required Packages")
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'tqdm': 'tqdm',
        'tensorboard': 'TensorBoard',
    }
    
    all_installed = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print_success(f"{name} is installed")
        except ImportError:
            print_error(f"{name} is NOT installed")
            all_installed = False
    
    return all_installed


def check_mfct_gan():
    """Check if MFCT-GAN can be imported"""
    print_header("MFCT-GAN Package")
    
    try:
        import mfct_gan
        print_success("MFCT-GAN package is importable")
        
        # Check key components
        from mfct_gan import (
            MFCT_GAN_Generator,
            PatchDiscriminator3D,
            MFCT_GAN_Loss,
            MFCT_GAN_Trainer,
        )
        print_success("All core components are available")
        
        print_info(f"Package version: {mfct_gan.__version__}")
        return True
    except ImportError as e:
        print_error(f"Cannot import MFCT-GAN: {e}")
        return False


def check_gpu():
    """Check GPU availability"""
    print_header("GPU Support")
    
    try:
        import torch
        if torch.cuda.is_available():
            print_success("CUDA GPU is available")
            print_info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print_info(f"CUDA Version: {torch.version.cuda}")
            print_info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print_info("CUDA GPU is NOT available - CPU training will be used (slower)")
            return False
    except Exception as e:
        print_error(f"Error checking GPU: {e}")
        return False


def test_forward_pass():
    """Test forward pass of the model"""
    print_header("Model Forward Pass Test")
    
    try:
        import torch
        from mfct_gan import MFCT_GAN_Generator, PatchDiscriminator3D
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create models
        generator = MFCT_GAN_Generator(base_channels=32).to(device)
        discriminator = PatchDiscriminator3D(base_channels=64).to(device)
        
        print_info("Models created successfully")
        
        # Test forward pass
        x_ray1 = torch.randn(1, 1, 128, 128).to(device)
        x_ray2 = torch.randn(1, 1, 128, 128).to(device)
        
        with torch.no_grad():
            ct_volume = generator(x_ray1, x_ray2)
            disc_output = discriminator(ct_volume)
        
        assert ct_volume.shape == (1, 1, 128, 128, 128), "Invalid generator output shape"
        assert disc_output.shape == (1, 1, 8, 8, 8), "Invalid discriminator output shape"
        
        print_success("Generator forward pass successful")
        print_info(f"  Input shapes: X-ray1 {x_ray1.shape}, X-ray2 {x_ray2.shape}")
        print_info(f"  Output shape: CT volume {ct_volume.shape}")
        print_success("Discriminator forward pass successful")
        print_info(f"  Output shape: {disc_output.shape}")
        
        return True
    except Exception as e:
        print_error(f"Forward pass test failed: {e}")
        return False


def count_parameters():
    """Count model parameters"""
    print_header("Model Parameters")
    
    try:
        from mfct_gan import MFCT_GAN_Generator, PatchDiscriminator3D
        
        generator = MFCT_GAN_Generator(base_channels=32)
        discriminator = PatchDiscriminator3D(base_channels=64)
        
        gen_params = sum(p.numel() for p in generator.parameters())
        disc_params = sum(p.numel() for p in discriminator.parameters())
        total_params = gen_params + disc_params
        
        print_info(f"Generator parameters: {gen_params:,}")
        print_info(f"Discriminator parameters: {disc_params:,}")
        print_info(f"Total parameters: {total_params:,}")
        
        print_success("Parameter counting successful")
        return True
    except Exception as e:
        print_error(f"Parameter counting failed: {e}")
        return False


def run_diagnostics():
    """Run all diagnostic checks"""
    print_header("MFCT-GAN Installation Diagnostic")
    
    checks = [
        ("Python Version", check_python),
        ("Required Packages", check_packages),
        ("MFCT-GAN Import", check_mfct_gan),
        ("GPU Support", check_gpu),
        ("Model Parameters", count_parameters),
        ("Forward Pass Test", test_forward_pass),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print_header("Diagnostic Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    print_info(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print_success("All checks passed! MFCT-GAN is ready to use.")
        print_info("\nNext steps:")
        print_info("1. Read QUICKSTART.md for quick setup")
        print_info("2. Run: python examples.py --test")
        print_info("3. Start training: python train.py --use_synthetic")
        return True
    else:
        print_error("Some checks failed. Please see above for details.")
        print_info("\nTroubleshooting:")
        print_info("1. Install missing packages: pip install -r requirements.txt")
        print_info("2. Check Python version: python --version")
        print_info("3. Review documentation in README.md")
        return False


def print_quick_commands():
    """Print quick reference commands"""
    print_header("Quick Reference Commands")
    
    commands = [
        ("Test installation", "python examples.py --test"),
        ("Run examples", "python examples.py"),
        ("Train with synthetic data", "python train.py --use_synthetic --num_epochs 10"),
        ("Train with real data", "python train.py --data_dir ./data --batch_size 8"),
        ("View TensorBoard", "tensorboard --logdir ./runs"),
        ("Check Python version", "python --version"),
        ("Update pip packages", "pip install -r requirements.txt"),
    ]
    
    for description, command in commands:
        print_info(f"{description}:")
        print(f"  $ {command}\n")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("  MFCT-GAN Getting Started")
    print("="*70)
    
    # Run diagnostics
    success = run_diagnostics()
    
    # Print quick commands
    print_quick_commands()
    
    # Print documentation guide
    print_header("Documentation Guide")
    print_info("Start with one of these based on your needs:")
    print_info("1. QUICKSTART.md - 5 minute setup guide")
    print_info("2. README.md - Complete documentation")
    print_info("3. API_REFERENCE.md - Detailed API reference")
    print_info("4. INDEX.md - Project navigation guide")
    
    print("\n" + "="*70)
    if success:
        print("  ✓ Installation successful! You're ready to go.")
    else:
        print("  ✗ Installation incomplete. Please fix the issues above.")
    print("="*70 + "\n")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
