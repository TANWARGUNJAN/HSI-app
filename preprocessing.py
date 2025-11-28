import numpy as np
from sklearn.model_selection import train_test_split
import gc

def create_patches(data, gt, patch_size=11, remove_zero_labels=True, sample_percentage=100):
    """
    Create patches from hyperspectral image with memory optimization
    
    Args:
        data: Hyperspectral image (H, W, Bands)
        gt: Ground truth labels (H, W)
        patch_size: Size of patches (default: 11)
        remove_zero_labels: Remove background pixels
        sample_percentage: Percentage of pixels to use (1-100)
    
    Returns:
        patches: Array of patches (N, patch_size, patch_size, bands)
        labels: Corresponding labels
    """
    height, width, bands = data.shape
    pad = patch_size // 2
    
    print(f"Input data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    
    # Normalize data first to reduce memory (convert to float32)
    data_normalized = data.astype(np.float32)
    
    # Simple normalization per band
    for i in range(bands):
        band = data_normalized[:, :, i]
        min_val = np.min(band)
        max_val = np.max(band)
        if max_val > min_val:
            data_normalized[:, :, i] = (band - min_val) / (max_val - min_val)
    
    # Pad the normalized data
    padded_data = np.pad(data_normalized, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    # Get valid pixel indices
    if remove_zero_labels:
        indices = np.argwhere(gt > 0)
    else:
        indices = np.argwhere(gt >= 0)
    
    labels = gt[indices[:, 0], indices[:, 1]]
    
    print(f"Total valid pixels: {len(indices)}")
    
    # Sample pixels if needed to reduce memory
    if sample_percentage < 100:
        n_samples = int(len(indices) * sample_percentage / 100)
        # Stratified sampling to keep class balance
        unique_labels = np.unique(labels)
        sampled_indices = []
        sampled_labels = []
        
        for label in unique_labels:
            label_mask = labels == label
            label_indices = indices[label_mask]
            n_label_samples = max(1, int(len(label_indices) * sample_percentage / 100))
            
            if len(label_indices) > 0:
                sample_idx = np.random.choice(len(label_indices), 
                                             min(n_label_samples, len(label_indices)), 
                                             replace=False)
                sampled_indices.append(label_indices[sample_idx])
                sampled_labels.append(np.full(len(sample_idx), label))
        
        indices = np.vstack(sampled_indices)
        labels = np.concatenate(sampled_labels)
        
        print(f"Sampled to {len(indices)} pixels ({sample_percentage}%)")
    
    # Calculate memory requirement
    memory_gb = (len(indices) * patch_size * patch_size * bands * 4) / (1024**3)  # 4 bytes for float32
    print(f"Estimated memory needed: {memory_gb:.2f} GB")
    
    if memory_gb > 4:
        print("⚠️ WARNING: Large memory requirement! Consider:")
        print("   - Reducing patch size")
        print("   - Reducing sample_percentage")
        print("   - Using smaller batch sizes during training")
    
    # Extract patches
    patches = np.zeros((len(indices), patch_size, patch_size, bands), dtype=np.float32)
    
    for i, (x, y) in enumerate(indices):
        patch = padded_data[x:x+patch_size, y:y+patch_size, :]
        patches[i] = patch
        
        if (i + 1) % 10000 == 0:
            print(f"Extracted {i+1}/{len(indices)} patches")
    
    print(f"Final patches shape: {patches.shape}")
    print(f"Final patches memory: {patches.nbytes / (1024**3):.2f} GB")
    
    # IMPORTANT: Adjust labels to start from 0 for PyTorch
    # Ground truth labels are usually 1,2,3... but PyTorch needs 0,1,2...
    unique_labels = np.unique(labels)
    print(f"Original label range: {unique_labels.min()} to {unique_labels.max()}")
    
    if unique_labels.min() > 0:
        labels = labels - unique_labels.min()
        print(f"Adjusted labels to: 0 to {labels.max()}")
    
    # Clean up
    del padded_data, data_normalized
    gc.collect()
    
    return patches, labels


def split_dataset(patches, labels, test_ratio=0.3):
    """
    Split dataset into train and test sets
    
    Args:
        patches: Patch array
        labels: Label array  
        test_ratio: Ratio for test set (default: 0.3 = 30% test, 70% train)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"\nSplitting dataset...")
    print(f"Input shape: {patches.shape}")
    print(f"Test ratio: {test_ratio} (Test: {test_ratio*100:.0f}%, Train: {(1-test_ratio)*100:.0f}%)")
    
    # Ensure float32 for memory efficiency
    if patches.dtype != np.float32:
        print(f"Converting from {patches.dtype} to float32...")
        patches = patches.astype(np.float32)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            patches, 
            labels,
            test_size=test_ratio,
            random_state=42,
            stratify=labels
        )
        
        print(f"Train set: {X_train.shape}, {X_train.nbytes / (1024**3):.2f} GB")
        print(f"Test set: {X_test.shape}, {X_test.nbytes / (1024**3):.2f} GB")
        
        # Clean up original arrays
        del patches, labels
        gc.collect()
        
        return X_train, X_test, y_train, y_test
        
    except MemoryError as e:
        print(f"❌ Memory Error: {e}")
        print("Suggestions:")
        print("1. Reduce patch_size (try 11 or 9 instead of 19)")
        print("2. Reduce sample_percentage in create_patches()")
        print("3. Close other applications to free RAM")
        raise


def apply_pca(data, n_components=30):
    """
    Apply PCA to reduce spectral dimensions
    Useful for datasets with many bands
    
    Args:
        data: Hyperspectral image (H, W, Bands)
        n_components: Number of principal components
    
    Returns:
        Transformed data (H, W, n_components)
    """
    from sklearn.decomposition import PCA
    
    print(f"Applying PCA: {data.shape[2]} -> {n_components} bands")
    
    h, w, bands = data.shape
    data_reshaped = data.reshape(-1, bands)
    
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_reshaped)
    data_pca = data_pca.reshape(h, w, n_components)
    
    print(f"Explained variance: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    return data_pca.astype(np.float32)


# Memory-efficient data generator (optional - for very large datasets)
class PatchGenerator:
    """
    Generator for loading patches in batches
    Use this if dataset is too large for memory
    """
    def __init__(self, data, gt, indices, patch_size, batch_size=32):
        self.data = data
        self.gt = gt
        self.indices = indices
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.pad = patch_size // 2
        self.padded_data = np.pad(data, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='reflect')
        
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        patches = []
        labels = []
        
        for x, y in batch_indices:
            patch = self.padded_data[x:x+self.patch_size, y:y+self.patch_size, :]
            patches.append(patch)
            labels.append(self.gt[x, y])
        
        return np.array(patches, dtype=np.float32), np.array(labels)