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
    """Transition block to convert 2D features to 3D by duplicating feature maps
    As specified in MFCT-GAN paper: "expanding the two dimensional to three-dimensional 
    by duplicating the feature maps"
    """
    def __init__(self, in_channels=256, spatial_size=16, output_channels=128):
        super(TransitionBlock, self).__init__()
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.output_channels = output_channels
        
        # Conv layer to adjust channels if needed
        if in_channels != output_channels:
            self.channel_adjust = nn.Conv2d(in_channels, output_channels, kernel_size=1)
        else:
            self.channel_adjust = None

    def forward(self, x):
        """
        Expand 2D features to 3D by duplicating along depth dimension
        Args:
            x: (B, C, H, W) 2D features
        Returns:
            x: (B, C', D, H, W) 3D features where D=H=W
        """
        # Adjust channels if needed
        if self.channel_adjust is not None:
            x = self.channel_adjust(x)
        
        # Expand 2D -> 3D by duplicating along depth dimension
        # (B, C, H, W) -> (B, C, 1, H, W) -> (B, C, D, H, W)
        x = x.unsqueeze(2).repeat(1, 1, self.spatial_size, 1, 1)
        
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
    Flexible output size based on target_size parameter
    Reduces channels BEFORE upsampling for memory efficiency
    """
    def __init__(self, in_channels=128, base_channels=64, target_size=128):
        super(Decoder3D, self).__init__()
        
        self.target_size = target_size
        # Calculate number of upsampling stages needed (from 16 to target_size)
        import math
        self.num_upsamples = int(math.log2(target_size // 16))
        
        # Build decoder layers dynamically with U-Net skip connections
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        # Expected skip connection channels (from encoder)
        # skip[0]: base_channels * 2, skip[1]: base_channels, skip[2]: base_channels // 2
        skip_channels = [base_channels * 2, base_channels, base_channels // 2]
        
        for i in range(self.num_upsamples):
            next_channels = max(base_channels // (2 ** i), 4)
            
            # Reduce channels before concatenation
            reduce = nn.Sequential(
                nn.Conv3d(current_channels, next_channels, kernel_size=1),
                nn.BatchNorm3d(next_channels),
                nn.ReLU(inplace=True)
            )
            basic3d = Basic3DBlock(next_channels, next_channels)
            
            # 1x1x1 conv to reduce concatenated channels back after skip connection
            # After concat: next_channels + skip_channels[i] -> next_channels
            if i < len(skip_channels):
                skip_reduce = nn.Sequential(
                    nn.Conv3d(next_channels + skip_channels[i], next_channels, kernel_size=1),
                    nn.BatchNorm3d(next_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                skip_reduce = None
            
            self.layers.append(nn.ModuleDict({
                'reduce': reduce,
                'basic3d': basic3d,
                'skip_reduce': skip_reduce
            }))
            
            current_channels = next_channels
        
        # Final convolution to get single channel output
        self.final = nn.Sequential(
            nn.Conv3d(current_channels, 1, kernel_size=1),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, x, skip_connections_3d=None):
        # Progressive upsampling with U-Net style skip connections (concatenation)
        for i, layer_dict in enumerate(self.layers):
            x = layer_dict['reduce'](x)
            x = layer_dict['basic3d'](x)
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
            
            # Concatenate skip connections if provided (U-Net style)
            if skip_connections_3d is not None and i < len(skip_connections_3d) and layer_dict['skip_reduce'] is not None:
                skip_3d = skip_connections_3d[i]
                # Resize skip connection to match current feature map spatial size
                if skip_3d.shape[2:] != x.shape[2:]:
                    skip_3d = F.interpolate(skip_3d, size=x.shape[2:], mode='trilinear', align_corners=False)
                
                # Concatenate along channel dimension (U-Net standard)
                x = torch.cat([x, skip_3d], dim=1)
                
                # Reduce channels back using 1x1x1 convolution
                x = layer_dict['skip_reduce'](x)
        
        # Final convolution
        x = self.final(x)
        
        # Ensure exact target size (in case of rounding issues)
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size, self.target_size), 
                            mode='trilinear', align_corners=False)
        
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
    def __init__(self, base_channels=32, ct_volume_size=128):
        super(MFCT_GAN_Generator, self).__init__()
        self.base_channels = base_channels
        self.ct_volume_size = ct_volume_size

        # Dual-parallel encoders
        self.dual_encoder = DualParallelEncoder(base_channels=base_channels)

        # Transition blocks for 2D to 3D conversion
        self.transition1 = TransitionBlock(
            in_channels=base_channels * 8, spatial_size=16, output_channels=base_channels * 4
        )
        self.transition2 = TransitionBlock(
            in_channels=base_channels * 8, spatial_size=16, output_channels=base_channels * 4
        )

        # Transition blocks for skip connections (2D to 3D)
        self.skip_transition1 = TransitionBlock(base_channels * 4, 32, base_channels * 2)
        self.skip_transition2 = TransitionBlock(base_channels * 2, 64, base_channels)
        self.skip_transition3 = TransitionBlock(base_channels, 128, base_channels // 2)
        
        # Skip Connection Modification modules for each skip level (after transition)
        self.scm_skip1 = SkipConnectionModification(in_channels=base_channels * 2, spatial_size=32)
        self.scm_skip2 = SkipConnectionModification(in_channels=base_channels, spatial_size=64)
        self.scm_skip3 = SkipConnectionModification(in_channels=base_channels // 2, spatial_size=128)
        
        # Feature fusion and 3D decoder
        self.decoder_3d = Decoder3D(
            in_channels=base_channels * 4, 
            base_channels=base_channels,
            target_size=ct_volume_size
        )

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
        features_3d_1 = self.transition1(bn1)
        features_3d_2 = self.transition2(bn2)

        # Expand spatial dimensions to match initial decoder input size (16x16x16)
        # The decoder will then progressively upsample to ct_volume_size
        initial_size = 16
        features_3d_1 = F.interpolate(
            features_3d_1, size=(initial_size, initial_size, initial_size), mode="trilinear", align_corners=False
        )
        features_3d_2 = F.interpolate(
            features_3d_2, size=(initial_size, initial_size, initial_size), mode="trilinear", align_corners=False
        )

        # Convert skip connections from 2D to 3D using SCM with second X-ray as weight map
        skip1_3d = self.skip_transition1(skip1[0])  # (B, C*4, 32, 32) -> (B, C*2, 32, 32, 32)
        skip1_3d = self.scm_skip1(skip1_3d, x_ray2)
        
        skip2_3d = self.skip_transition2(skip1[1])  # (B, C*2, 64, 64) -> (B, C, 64, 64, 64)
        skip2_3d = self.scm_skip2(skip2_3d, x_ray2)
        
        skip3_3d = self.skip_transition3(skip1[2])  # (B, C, 128, 128) -> (B, C//2, 128, 128, 128)
        skip3_3d = self.scm_skip3(skip3_3d, x_ray2)
        
        skip_connections_3d = [skip1_3d, skip2_3d, skip3_3d]

        # Feature fusion by averaging
        fused_features = (features_3d_1 + features_3d_2) / 2.0

        # 3D upsampling decoder with skip connections
        ct_volume = self.decoder_3d(fused_features, skip_connections_3d)

        return ct_volume


class PatchDiscriminator3D(nn.Module):
    """PatchGAN Discriminator as specified in MFCT-GAN paper
    Uses kernel size 5×3 three times, followed by kernel size 5×1
    """
    def __init__(self, in_channels=1, base_channels=64, volume_size=128):
        super(PatchDiscriminator3D, self).__init__()
        
        # As specified in paper: kernel size 5×3 (interpreted as 5×5×5 with stride 3)
        # followed by kernel size 5×1 (interpreted as 5×5×5 with stride 1)
        
        def conv3d_block(in_ch, out_ch, kernel_size=5, stride=2, padding=2):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        
        # Three layers with kernel 5, stride 2 (paper says "5×3" likely meaning stride 3)
        self.layer1 = conv3d_block(in_channels, base_channels, kernel_size=5, stride=2, padding=2)
        self.layer2 = conv3d_block(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2)
        self.layer3 = conv3d_block(base_channels * 2, base_channels * 4, kernel_size=5, stride=2, padding=2)
        
        # One layer with kernel 5, stride 1 (paper says "5×1")
        self.layer4 = conv3d_block(base_channels * 4, base_channels * 8, kernel_size=5, stride=1, padding=2)
        
        # Final patch discriminator output
        self.final = nn.Conv3d(base_channels * 8, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        """
        Args:
            x: 3D volume (B, 1, D, H, W)
        Returns:
            Discriminator output (patch-level predictions)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)
        return x
