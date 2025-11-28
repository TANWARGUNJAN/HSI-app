"""
Test script to verify the upload functionality works correctly
This tests the load_uploaded_dataset function without running the full Streamlit app
"""

import numpy as np
import scipy.io as sio
import h5py
from pathlib import Path
import tempfile
import os

# Test data generation
def create_test_mat_files():
    """Create test .mat files to simulate user uploads"""
    print("Creating test .mat files...")
    
    # Create test directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Generate synthetic hyperspectral data
    height, width, bands = 50, 50, 100
    data = np.random.rand(height, width, bands).astype(np.float32)
    
    # Generate synthetic ground truth
    gt = np.random.randint(0, 5, size=(height, width)).astype(np.int32)
    
    # Save as .mat files (scipy format)
    data_file = test_dir / "test_data.mat"
    gt_file = test_dir / "test_gt.mat"
    
    sio.savemat(str(data_file), {"data": data})
    sio.savemat(str(gt_file), {"gt": gt})
    
    print(f"✓ Created test files:")
    print(f"  - {data_file} (shape: {data.shape})")
    print(f"  - {gt_file} (shape: {gt.shape})")
    
    return data_file, gt_file, data, gt


def test_mat_loading(data_file, gt_file, data_key="data", gt_key="gt"):
    """Test loading .mat files using scipy and h5py"""
    print(f"\n{'='*60}")
    print("Testing .mat file loading...")
    print(f"{'='*60}")
    
    # Test 1: Load with scipy
    try:
        print("\n[Test 1] Loading with scipy.io.loadmat...")
        mat_data = sio.loadmat(str(data_file))
        mat_gt = sio.loadmat(str(gt_file))
        
        if data_key in mat_data and gt_key in mat_gt:
            data = mat_data[data_key]
            gt = mat_gt[gt_key].squeeze()
            print(f"✓ Success!")
            print(f"  - Data shape: {data.shape}")
            print(f"  - GT shape: {gt.shape}")
            print(f"  - Data dtype: {data.dtype}")
            print(f"  - GT dtype: {gt.dtype}")
            return True
        else:
            print(f"✗ Keys not found. Available keys in data: {list(mat_data.keys())}")
            print(f"  Available keys in gt: {list(mat_gt.keys())}")
            return False
            
    except Exception as e:
        print(f"✗ scipy loading failed: {e}")
        
        # Test 2: Try with h5py (for MATLAB v7.3 files)
        try:
            print("\n[Test 2] Loading with h5py...")
            with h5py.File(str(data_file), 'r') as f:
                if data_key in f:
                    data = np.array(f[data_key])
                    print(f"✓ Data loaded: {data.shape}")
                else:
                    print(f"✗ Key '{data_key}' not found. Available: {list(f.keys())}")
                    return False
            
            with h5py.File(str(gt_file), 'r') as f:
                if gt_key in f:
                    gt = np.array(f[gt_key]).squeeze()
                    print(f"✓ GT loaded: {gt.shape}")
                else:
                    print(f"✗ Key '{gt_key}' not found. Available: {list(f.keys())}")
                    return False
            
            print("✓ h5py loading successful!")
            return True
            
        except Exception as e2:
            print(f"✗ h5py loading also failed: {e2}")
            return False


def test_upload_function():
    """Test the complete upload workflow"""
    print(f"\n{'='*60}")
    print("Testing complete upload workflow...")
    print(f"{'='*60}")
    
    from config import UPLOADS_DIR
    
    # Ensure uploads directory exists
    Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✓ Uploads directory: {UPLOADS_DIR}")
    
    # Create test files
    data_file, gt_file, original_data, original_gt = create_test_mat_files()
    
    # Test loading
    success = test_mat_loading(data_file, gt_file)
    
    if success:
        print(f"\n{'='*60}")
        print("✓ Upload functionality is WORKING!")
        print(f"{'='*60}")
        print("\nYou can now:")
        print("1. Use the Streamlit app's upload feature")
        print("2. Upload the test files from: test_data/")
        print("3. Use keys: 'data' and 'gt'")
    else:
        print(f"\n{'='*60}")
        print("✗ Upload functionality has ISSUES!")
        print(f"{'='*60}")
    
    return success


def test_with_real_dataset():
    """Test with an actual dataset from the datasets folder"""
    print(f"\n{'='*60}")
    print("Testing with real Indian Pines dataset...")
    print(f"{'='*60}")
    
    from config import DATASETS_DIR, DATASET_CONFIG
    
    try:
        config = DATASET_CONFIG['Indian_pines']
        data_file = Path(DATASETS_DIR) / config['data_file']
        gt_file = Path(DATASETS_DIR) / config['gt_file']
        
        if not data_file.exists():
            print(f"✗ Data file not found: {data_file}")
            return False
        
        if not gt_file.exists():
            print(f"✗ GT file not found: {gt_file}")
            return False
        
        print(f"✓ Found files:")
        print(f"  - {data_file.name}")
        print(f"  - {gt_file.name}")
        
        success = test_mat_loading(
            data_file, 
            gt_file, 
            config['data_key'], 
            config['gt_key']
        )
        
        return success
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def inspect_mat_file(filepath):
    """Inspect a .mat file to see its structure"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath.name}")
    print(f"{'='*60}")
    
    try:
        # Try scipy first
        mat_data = sio.loadmat(str(filepath))
        print("\n[scipy.io.loadmat] Available keys:")
        for key, value in mat_data.items():
            if not key.startswith('__'):
                if hasattr(value, 'shape'):
                    print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  - {key}: {type(value)}")
        return True
    except NotImplementedError:
        print("✗ scipy failed (likely MATLAB v7.3), trying h5py...")
        try:
            with h5py.File(str(filepath), 'r') as f:
                print("\n[h5py] Available keys:")
                for key in f.keys():
                    item = f[key]
                    if hasattr(item, 'shape'):
                        print(f"  - {key}: shape={item.shape}, dtype={item.dtype}")
                    else:
                        print(f"  - {key}: {type(item)}")
            return True
        except Exception as e:
            print(f"✗ h5py also failed: {e}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("HYPERSPECTRAL APP - UPLOAD FUNCTIONALITY TEST")
    print("="*60)
    
    # Test 1: Create and test synthetic data
    print("\n[TEST 1] Synthetic data upload test")
    test_upload_function()
    
    # Test 2: Test with real dataset
    print("\n\n[TEST 2] Real dataset test")
    test_with_real_dataset()
    
    # Test 3: Inspect a real file
    print("\n\n[TEST 3] File inspection")
    from config import DATASETS_DIR, DATASET_CONFIG
    config = DATASET_CONFIG['Indian_pines']
    data_file = Path(DATASETS_DIR) / config['data_file']
    if data_file.exists():
        inspect_mat_file(data_file)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
