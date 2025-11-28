import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSN(nn.Module):
    """
    HybridSN: Exploring 3D-2D CNN Feature Hierarchy for Hyperspectral Image Classification
    Adapted to work with any patch size
    """
    def __init__(self, num_classes, num_bands, patch_size):
        super(HybridSN, self).__init__()
        
        self.num_classes = num_classes
        self.num_bands = num_bands
        self.patch_size = patch_size
        
        # 3D Convolutional Layers
        self.conv3d_1 = nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(0, 1, 1))
        self.bn3d_1 = nn.BatchNorm3d(8)
        
        self.conv3d_2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(0, 1, 1))
        self.bn3d_2 = nn.BatchNorm3d(16)
        
        self.conv3d_3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(0, 1, 1))
        self.bn3d_3 = nn.BatchNorm3d(32)
        
        # Calculate the size after 3D convolutions
        # After conv3d_1: bands -> bands - 7 + 1 = bands - 6
        # After conv3d_2: bands - 6 -> bands - 6 - 5 + 1 = bands - 10
        # After conv3d_3: bands - 10 -> bands - 10 - 3 + 1 = bands - 12
        self.conv3d_output_bands = num_bands - 12
        
        # Reshape from 3D to 2D
        # Input channels after reshape: 32 * conv3d_output_bands
        self.conv2d_input_channels = 32 * self.conv3d_output_bands
        
        # 2D Convolutional Layers
        self.conv2d_1 = nn.Conv2d(self.conv2d_input_channels, 64, kernel_size=(3, 3), padding=1)
        self.bn2d_1 = nn.BatchNorm2d(64)
        
        # Calculate size after 2D convolutions and pooling
        # After conv2d_1 with padding=1: patch_size stays same
        # No pooling, so spatial dimensions remain: patch_size x patch_size
        
        # Flatten size calculation
        self.flatten_size = 64 * patch_size * patch_size
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(128, num_classes)
        
        print(f"Model initialized:")
        print(f"  - Input: ({patch_size}, {patch_size}, {num_bands})")
        print(f"  - 3D output bands: {self.conv3d_output_bands}")
        print(f"  - 2D input channels: {self.conv2d_input_channels}")
        print(f"  - Flatten size: {self.flatten_size}")
        print(f"  - Output classes: {num_classes}")
    
    def forward(self, x):
        # Input shape: (batch, patch_size, patch_size, bands)
        # Reshape to: (batch, 1, bands, patch_size, patch_size) for 3D conv
        x = x.permute(0, 3, 1, 2).unsqueeze(1)  # (batch, 1, bands, H, W)
        
        # 3D Convolutional layers
        x = F.relu(self.bn3d_1(self.conv3d_1(x)))
        x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        x = F.relu(self.bn3d_3(self.conv3d_3(x)))
        
        # Reshape from 5D to 4D for 2D convolutions
        # From: (batch, 32, conv3d_output_bands, H, W)
        # To: (batch, 32*conv3d_output_bands, H, W)
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.conv2d_input_channels, self.patch_size, self.patch_size)
        
        # 2D Convolutional layer
        x = F.relu(self.bn2d_1(self.conv2d_1(x)))
        
        # Flatten
        x = x.reshape(batch_size, -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class SimpleCNN(nn.Module):
    """
    Simpler CNN alternative if HybridSN still has issues
    Works with any patch size and number of bands
    """
    def __init__(self, num_classes, num_bands, patch_size):
        super(SimpleCNN, self).__init__()
        
        self.num_classes = num_classes
        self.num_bands = num_bands
        self.patch_size = patch_size
        
        # 3D Convolution to extract spectral-spatial features
        self.conv3d_1 = nn.Conv3d(1, 16, kernel_size=(7, 3, 3), padding=(0, 1, 1))
        self.bn3d_1 = nn.BatchNorm3d(16)
        
        self.conv3d_2 = nn.Conv3d(16, 32, kernel_size=(5, 3, 3), padding=(0, 1, 1))
        self.bn3d_2 = nn.BatchNorm3d(32)
        
        # After 3D convolutions
        bands_after_3d = num_bands - 10  # 7-1 + 5-1 = 10
        
        # 2D Convolutions
        self.conv2d_1 = nn.Conv2d(32 * bands_after_3d, 64, kernel_size=3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(64)
        
        # Global Average Pooling to handle any spatial size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        print(f"SimpleCNN initialized: {patch_size}x{patch_size}x{num_bands} -> {num_classes} classes")
    
    def forward(self, x):
        # Input: (batch, H, W, bands)
        x = x.permute(0, 3, 1, 2).unsqueeze(1)  # (batch, 1, bands, H, W)
        
        # 3D convolutions
        x = F.relu(self.bn3d_1(self.conv3d_1(x)))
        x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        
        # Reshape to 2D
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.patch_size, self.patch_size)
        
        # 2D convolution
        x = F.relu(self.bn2d_1(self.conv2d_1(x)))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# Helper function to create the right model
def create_model(model_name, num_classes, num_bands, patch_size):
    """
    Factory function to create models
    
    Args:
        model_name: 'hybridsn' or 'simplecnn'
        num_classes: Number of output classes
        num_bands: Number of spectral bands
        patch_size: Spatial patch size
    """
    if model_name.lower() == 'hybridsn':
        return HybridSN(num_classes, num_bands, patch_size)
    elif model_name.lower() == 'simplecnn':
        return SimpleCNN(num_classes, num_bands, patch_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")