"""
MFCT-GAN: Multi-information Fusion Network for 3D CT Reconstruction
Reconstructs 3D CT volumes from two biplanar orthogonal 2D X-ray images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    """Basic 2D convolutional block with batch norm and ReLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class ConvBlock3D(nn.Module):
    """Basic 3D convolutional block with batch norm and ReLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class DenseBlock2D(nn.Module):
    """Dense Block for 2D feature extraction"""
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(DenseBlock2D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                ConvBlock2D(in_channels + i * growth_rate, growth_rate)
            )
        self.growth_rate = growth_rate
        self.num_layers = num_layers

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)


class MultiChannelResidualDenseBlock(nn.Module):
    """Multi-channels Residual Dense Block (MRDB) for initial feature extraction
    Based on the architecture diagram with parallel paths using different kernel sizes (7, 3, 1)
    """
    def __init__(self, in_channels, out_channels, growth_rate=32, num_layers=4):
        super(MultiChannelResidualDenseBlock, self).__init__()
        
        # Multi-resolution parallel paths with different kernel sizes
        # Path 1: Large kernel (7x7)
        self.conv_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=7, padding=3),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        
        # Path 2: Medium kernel (3x3)
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        
        # Path 3: Small kernel (1x1)
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=1, padding=0),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        
        # Dense connections
        self.dense_block = DenseBlock2D(growth_rate * 3, growth_rate, num_layers)
        dense_output_channels = growth_rate * 3 + num_layers * growth_rate
        
        # 1x1 convolution to adjust channels
        self.conv_adjust = nn.Conv2d(dense_output_channels, out_channels, 1)
        self.norm_final = nn.BatchNorm2d(out_channels)
        
        # Residual connection adjustment if needed
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        # Multi-resolution paths
        out_7 = self.conv_7x7(x)
        out_3 = self.conv_3x3(x)
        out_1 = self.conv_1x1(x)
        
        # Concatenate multi-resolution features
        out = torch.cat([out_7, out_3, out_1], dim=1)
        
        # Dense block processing
        out = self.dense_block(out)
        
        # Adjust channels and add residual
        out = self.conv_adjust(out)
        out = self.norm_final(out)
        out = out + residual
        
        return out


class Encoder2D(nn.Module):
    """2D Encoder for parallel feature extraction from single X-ray image"""
    def __init__(self, in_channels=1, base_channels=32):
        super(Encoder2D, self).__init__()
        self.layer1 = nn.Sequential(
            ConvBlock2D(in_channels, base_channels),
            MultiChannelResidualDenseBlock(base_channels, base_channels),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.layer2 = nn.Sequential(
            ConvBlock2D(base_channels, base_channels * 2),
            MultiChannelResidualDenseBlock(base_channels * 2, base_channels * 2),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.layer3 = nn.Sequential(
            ConvBlock2D(base_channels * 2, base_channels * 4),
            MultiChannelResidualDenseBlock(base_channels * 4, base_channels * 4),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ConvBlock2D(base_channels * 4, base_channels * 8),
            MultiChannelResidualDenseBlock(base_channels * 8, base_channels * 8),
        )

    def forward(self, x):
        # Encoder with skip connections
        skip1 = self.layer1(x)
        x = self.pool1(skip1)

        skip2 = self.layer2(x)
        x = self.pool2(skip2)

        skip3 = self.layer3(x)
        x = self.pool3(skip3)

        bottleneck = self.bottleneck(x)

        return bottleneck, (skip3, skip2, skip1)


class TransitionBlock(nn.Module):
    """Transition block with fully connected layer to convert 2D features to 3D"""
    def __init__(self, in_channels=256, spatial_size=16, output_depth=128):
        super(TransitionBlock, self).__init__()
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.output_depth = output_depth

        # Flatten and reshape dimensions
        flattened_size = in_channels * spatial_size * spatial_size
        self.fc = nn.Linear(flattened_size, output_depth * 32 * 32 * 4)
        self.norm = nn.BatchNorm1d(output_depth * 32 * 32 * 4)

    def forward(self, x):
        batch_size = x.shape[0]
        # Flatten
        x = x.view(batch_size, -1)
        # Fully connected layer
        x = self.fc(x)
        x = self.norm(x)
        x = F.relu(x)
        # Reshape to 3D
        x = x.view(batch_size, self.output_depth, 32, 32, 4)
        return x


class SkipConnectionModification(nn.Module):
    """Skip Connection Modification (SCM) module
    Uses second X-ray image as weight map to correct 3D features
    Based on architecture diagram showing skip connections at encoder-decoder interface
    """
    def __init__(self, in_channels, spatial_size=16):
        super(SkipConnectionModification, self).__init__()
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        
        # Multi-layer processing for weight map (matching diagram structure)
        self.weight_conv = nn.Sequential(
            nn.Conv2d(1, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()  # Use sigmoid to get weight values in [0, 1]
        )
        
        # Additional 3D convolution for feature refinement
        self.refine_3d = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features_3d, weight_map):
        """
        Args:
            features_3d: 3D features from first X-ray (B, C, D, H, W)
            weight_map: Second X-ray image as weight map (B, 1, H, W)
        Returns:
            Weighted 3D features
        """
        batch_size, channels, depth, height, width = features_3d.shape

        # Resize weight map to match feature spatial dimensions if needed
        if weight_map.shape[-2:] != (height, width):
            weight_map = F.interpolate(weight_map, size=(height, width), mode='bilinear', align_corners=False)

        # Process weight map through convolutional layers
        processed_weight = self.weight_conv(weight_map)  # (B, C, H, W)

        # Expand weight map to 3D by repeating along depth dimension
        # This makes the dimension expansion more physically reasonable
        weight_3d = processed_weight.unsqueeze(2).expand(
            batch_size, channels, depth, height, width
        )

        # Apply weighted modulation (element-wise multiplication)
        weighted_features = features_3d * weight_3d
        
        # Refine with 3D convolution
        weighted_features = self.refine_3d(weighted_features)
        
        # Add residual connection
        weighted_features = weighted_features + features_3d

        return weighted_features


class Basic3DBlock(nn.Module):
    """Basic 3D block as shown in the architecture diagram
    Contains two 3D convolutions with different kernel sizes (3x3x3 and 1x1x1)
    """
    def __init__(self, in_channels, out_channels):
        super(Basic3DBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoder3D(nn.Module):
    """3D Decoder for upsampling and feature reconstruction
    Based on architecture diagram with Basic3D blocks and upsampling
    """
    def __init__(self, in_channels=256, base_channels=64):
        super(Decoder3D, self).__init__()

        # Decoder layer 1 with upsampling
        self.upsample1 = nn.ConvTranspose3d(
            in_channels, base_channels * 4, 
            kernel_size=4, stride=2, padding=1
        )
        self.basic3d_1 = Basic3DBlock(base_channels * 4, base_channels * 4)

        # Decoder layer 2 with upsampling
        self.upsample2 = nn.ConvTranspose3d(
            base_channels * 4, base_channels * 2, 
            kernel_size=4, stride=2, padding=1
        )
        self.basic3d_2 = Basic3DBlock(base_channels * 2, base_channels * 2)

        # Decoder layer 3 with upsampling
        self.upsample3 = nn.ConvTranspose3d(
            base_channels * 2, base_channels, 
            kernel_size=4, stride=2, padding=1
        )
        self.basic3d_3 = Basic3DBlock(base_channels, base_channels)

        # Final convolution to get single channel output
        self.final = nn.Sequential(
            nn.Conv3d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels // 2, 1, kernel_size=1, padding=0),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, x):
        # Layer 1: Upsample + Basic3D
        x = self.upsample1(x)
        x = self.basic3d_1(x)
        
        # Layer 2: Upsample + Basic3D
        x = self.upsample2(x)
        x = self.basic3d_2(x)
        
        # Layer 3: Upsample + Basic3D
        x = self.upsample3(x)
        x = self.basic3d_3(x)
        
        # Final convolution
        x = self.final(x)
        return x


class DualParallelEncoder(nn.Module):
    """Dual-parallel encoder for processing two orthogonal X-ray views"""
    def __init__(self, base_channels=32):
        super(DualParallelEncoder, self).__init__()
        self.encoder1 = Encoder2D(in_channels=1, base_channels=base_channels)
        self.encoder2 = Encoder2D(in_channels=1, base_channels=base_channels)

    def forward(self, x_ray1, x_ray2):
        """
        Args:
            x_ray1: First orthogonal X-ray view (B, 1, H, W)
            x_ray2: Second orthogonal X-ray view (B, 1, H, W)
        Returns:
            bottleneck_features1: Bottleneck features from encoder1
            bottleneck_features2: Bottleneck features from encoder2
            skip_connections1: Skip connections from encoder1
            skip_connections2: Skip connections from encoder2
        """
        bn1, skip1 = self.encoder1(x_ray1)
        bn2, skip2 = self.encoder2(x_ray2)
        return bn1, bn2, skip1, skip2


class MFCT_GAN_Generator(nn.Module):
    """MFCT-GAN Generator: Complete architecture for 3D CT reconstruction"""
    def __init__(self, base_channels=32):
        super(MFCT_GAN_Generator, self).__init__()
        self.base_channels = base_channels

        # Dual-parallel encoders
        self.dual_encoder = DualParallelEncoder(base_channels=base_channels)

        # Transition blocks for 2D to 3D conversion
        self.transition1 = TransitionBlock(
            in_channels=base_channels * 8, spatial_size=16, output_depth=128
        )
        self.transition2 = TransitionBlock(
            in_channels=base_channels * 8, spatial_size=16, output_depth=128
        )

        # Skip Connection Modification module
        self.scm = SkipConnectionModification(in_channels=128)

        # Feature fusion and 3D decoder
        self.decoder_3d = Decoder3D(in_channels=128, base_channels=base_channels * 2)

    def forward(self, x_ray1, x_ray2):
        """
        Args:
            x_ray1: First orthogonal X-ray view (B, 1, 128, 128)
            x_ray2: Second orthogonal X-ray view (B, 1, 128, 128)
        Returns:
            ct_volume: Reconstructed 3D CT volume (B, 1, 128, 128, 128)
        """
        # Dual parallel encoding
        bn1, bn2, skip1, skip2 = self.dual_encoder(x_ray1, x_ray2)

        # Convert 2D features to 3D through transition blocks
        features_3d_1 = self.transition1(bn1)  # (B, 128, 32, 32, 4)
        features_3d_2 = self.transition2(bn2)  # (B, 128, 32, 32, 4)

        # Expand spatial dimensions
        features_3d_1 = F.interpolate(
            features_3d_1, size=(128, 128, 128), mode="trilinear", align_corners=False
        )
        features_3d_2 = F.interpolate(
            features_3d_2, size=(128, 128, 128), mode="trilinear", align_corners=False
        )

        # Apply Skip Connection Modification using second X-ray as weight map
        features_3d_1 = self.scm(features_3d_1, x_ray2)

        # Feature fusion by averaging
        fused_features = (features_3d_1 + features_3d_2) / 2.0

        # 3D upsampling decoder
        ct_volume = self.decoder_3d(fused_features)

        return ct_volume


class PatchDiscriminator3D(nn.Module):
    """Modified PatchGAN Discriminator with 3D convolutional modules"""
    def __init__(self, in_channels=1, base_channels=64):
        super(PatchDiscriminator3D, self).__init__()

        def conv3d_block(in_ch, out_ch, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=stride, padding=padding),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.layer1 = conv3d_block(in_channels, base_channels, stride=2)
        self.layer2 = conv3d_block(base_channels, base_channels * 2, stride=2)
        self.layer3 = conv3d_block(base_channels * 2, base_channels * 4, stride=2)
        self.layer4 = conv3d_block(base_channels * 4, base_channels * 8, stride=2)

        # Patch discriminator output
        self.final = nn.Conv3d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        """
        Args:
            x: 3D volume (B, 1, 128, 128, 128)
        Returns:
            discriminator_output: Patch-wise discrimination scores
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)
        return x
