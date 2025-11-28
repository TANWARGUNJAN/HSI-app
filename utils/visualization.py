# FILE: utils/visualization.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np

def plot_false_color(data, bands_selection=None):
    """
    Create false color composite
    
    Args:
        data: Hyperspectral data (height, width, bands)
        bands_selection: Tuple of (r_band, g_band, b_band) indices
        
    Returns:
        fig: Matplotlib figure
    """
    if data is None or data.ndim != 3:
        raise ValueError("Invalid data format")
    
    height, width, num_bands = data.shape
    
    if bands_selection is None:
        r_band = min(num_bands - 1, int(num_bands * 0.7))
        g_band = min(num_bands - 1, int(num_bands * 0.5))
        b_band = min(num_bands - 1, int(num_bands * 0.3))
    else:
        r_band, g_band, b_band = bands_selection
    
    # Normalize each band
    data_normalized = data.astype(np.float32)
    for i in range(num_bands):
        band_min = data_normalized[:, :, i].min()
        band_max = data_normalized[:, :, i].max()
        if band_max - band_min > 0:
            data_normalized[:, :, i] = (data_normalized[:, :, i] - band_min) / (band_max - band_min)
        else:
            data_normalized[:, :, i] = 0
    
    # Create RGB composite
    rgb = np.stack([
        data_normalized[:, :, r_band],
        data_normalized[:, :, g_band],
        data_normalized[:, :, b_band]
    ], axis=2)
    
    # Clip to valid range
    rgb = np.clip(rgb, 0, 1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)
    ax.set_title('False Color Composite')
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def plot_ground_truth(gt, class_labels=None):
    """
    Plot ground truth map with proper colors
    
    Args:
        gt: Ground truth labels (height, width)
        class_labels: List of class names
        
    Returns:
        fig: Matplotlib figure
    """
    if gt is None:
        raise ValueError("Ground truth data is None")
    
    gt = gt.astype(np.int32)
    num_classes = int(gt.max())
    
    if class_labels is None:
        class_labels = [f'Class {i}' for i in range(num_classes + 1)]
    
    # Create custom colormap
    colors_list = plt.cm.tab20(np.linspace(0, 1, max(num_classes + 2, 20)))
    cmap = ListedColormap(colors_list)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot ground truth with discrete colors
    im = ax.imshow(gt, cmap=cmap, interpolation='nearest', vmin=0, vmax=num_classes)
    
    # Create legend with only labeled classes
    patches = []
    for i in range(num_classes + 1):
        if i < len(class_labels):
            label = class_labels[i]
        else:
            label = f'Class {i}'
        
        color = colors_list[i % len(colors_list)]
        patches.append(mpatches.Patch(color=color, label=label))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Class')
    
    ax.set_title('Ground Truth Map', fontsize=14, fontweight='bold')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    
    # Add legend
    ax.legend(handles=patches, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    return fig

def plot_spectral_signature(data, pixel_position, num_bands=None):
    """
    Plot spectral signature for a pixel
    
    Args:
        data: Hyperspectral data (height, width, bands)
        pixel_position: Tuple of (i, j) pixel position
        num_bands: Number of bands to plot (None = all)
        
    Returns:
        fig: Matplotlib figure
    """
    if data is None or data.ndim != 3:
        raise ValueError("Invalid data format")
    
    i, j = pixel_position
    
    # Bounds checking
    if i < 0 or i >= data.shape[0] or j < 0 or j >= data.shape[1]:
        raise ValueError(f"Pixel position ({i}, {j}) out of bounds")
    
    spectrum = data[i, j, :]
    
    if num_bands is not None:
        spectrum = spectrum[:num_bands]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(spectrum, linewidth=2, color='#2E86AB')
    ax.fill_between(range(len(spectrum)), spectrum, alpha=0.3, color='#2E86AB')
    ax.set_xlabel('Band Index', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.set_title(f'Spectral Signature at Pixel ({i}, {j})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_classification_map(predictions, title="Classification Map"):
    """
    Plot classification/prediction map
    
    Args:
        predictions: 2D array of class predictions
        title: Title for the plot
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    num_classes = int(predictions.max())
    colors_list = plt.cm.tab20(np.linspace(0, 1, max(num_classes + 1, 20)))
    cmap = ListedColormap(colors_list)
    
    im = ax.imshow(predictions, cmap=cmap, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Class')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig