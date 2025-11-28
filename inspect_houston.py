"""
Quick script to inspect Houston13.mat file structure
"""
import scipy.io as sio
import h5py
import numpy as np

filepath = "Houston13.mat"  # Adjust if file is in a different location

print("="*60)
print(f"Inspecting: {filepath}")
print("="*60)

# Try scipy first
try:
    print("\n[Method 1] Using scipy.io.loadmat...")
    mat_data = sio.loadmat(filepath)
    print("\n✓ Success! Available keys:")
    for key, value in mat_data.items():
        if not key.startswith('__'):
            if hasattr(value, 'shape'):
                print(f"  📦 {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  📦 {key}: {type(value)}")
except NotImplementedError:
    print("\n✗ scipy failed (MATLAB v7.3 format)")
    print("\n[Method 2] Using h5py...")
    try:
        with h5py.File(filepath, 'r') as f:
            print("\n✓ Success! Available keys:")
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  📦 {name}: shape={obj.shape}, dtype={obj.dtype}")
            
            f.visititems(print_structure)
    except Exception as e:
        print(f"\n✗ h5py also failed: {e}")
except FileNotFoundError:
    print(f"\n✗ File not found: {filepath}")
    print("Make sure the file is in the current directory or provide the full path")
except Exception as e:
    print(f"\n✗ Error: {e}")

print("\n" + "="*60)
print("INSTRUCTIONS:")
print("="*60)
print("Use the key names shown above in the Streamlit app's")
print("'Data key' and 'GT key' input fields.")
print("="*60)
