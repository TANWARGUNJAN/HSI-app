# FILE: utils/data_loader.py
import numpy as np
import scipy.io as sio
import h5py
from pathlib import Path
import streamlit as st

class DataLoader:
    """Load hyperspectral datasets"""
    
    def __init__(self, dataset_name):
        """
        Initialize data loader
        
        Args:
            dataset_name: Name of dataset ('Indian_pines', 'Pavia', 'PaviaU', 'Salinas', 'Chikusei')
        """
        self.dataset_name = dataset_name
        self.data = None
        self.gt = None
        self.height = 0
        self.width = 0
        self.num_bands = 0
    
    def load_data(self):
        """Load dataset from scipy or HDF5"""
        datasets_config = {
            'Indian_pines': {
                'data_keys': ['indian_pines', 'data', 'ori_data'],
                'gt_keys': ['indian_pines_gt', 'gt', 'ground_truth'],
                'filename': 'indian_pines.mat'
            },
            'Pavia': {
                'data_keys': ['pavia', 'data', 'ori_data'],
                'gt_keys': ['pavia_gt', 'gt', 'ground_truth'],
                'filename': 'pavia.mat'
            },
            'PaviaU': {
                'data_keys': ['paviaU', 'data', 'ori_data'],
                'gt_keys': ['paviaU_gt', 'gt', 'ground_truth'],
                'filename': 'paviaU.mat'
            },
            'Salinas': {
                'data_keys': ['salinas', 'data', 'ori_data'],
                'gt_keys': ['salinas_gt', 'gt', 'ground_truth'],
                'filename': 'salinas.mat'
            },
            'Chikusei': {
                'data_keys': ['chikusei', 'data', 'ori_data'],
                'gt_keys': ['chikusei_gt', 'gt', 'ground_truth'],
                'filename': 'chikusei.mat'
            }
        }
        
        if self.dataset_name not in datasets_config:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
        
        config = datasets_config[self.dataset_name]
        
        try:
            self._load_from_file(config)
            if self.data is None or self.data.size == 0:
                raise ValueError("Data is empty after loading")
        except Exception as e:
            st.warning(f"Dataset file not found. Using sample data for {self.dataset_name}")
            self._create_sample_data(self.dataset_name)
        
        # Update dimensions
        if self.data is not None:
            self.height = self.data.shape[0]
            self.width = self.data.shape[1]
            self.num_bands = self.data.shape[2] if self.data.ndim == 3 else 1
    
    def _load_from_file(self, config):
        """Try to load from file using scipy or h5py"""
        data_keys = config['data_keys']
        gt_keys = config['gt_keys']
        filename = config['filename']
        
        # Try scipy first
        try:
            mat_data = sio.loadmat(filename)
            self.data = self._find_array(mat_data, data_keys)
            self.gt = self._find_array(mat_data, gt_keys)
            if self.data is not None:
                return
        except (NotImplementedError, FileNotFoundError, KeyError):
            pass
        
        # Try HDF5 format
        try:
            with h5py.File(filename, 'r') as f:
                self.data = self._find_array_hdf5(f, data_keys)
                self.gt = self._find_array_hdf5(f, gt_keys)
            if self.data is not None:
                return
        except Exception:
            pass
        
        raise FileNotFoundError(f"Could not load {filename}")
    
    def _find_array(self, mat_dict, keys):
        """Find array in scipy mat dictionary by trying multiple keys"""
        for key in keys:
            if key in mat_dict:
                arr = mat_dict[key]
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    return arr
        
        # If no specific key found, get the first large array
        for key in mat_dict.keys():
            if not key.startswith('__'):
                arr = mat_dict[key]
                if isinstance(arr, np.ndarray) and arr.size > 0 and arr.ndim >= 2:
                    return arr
        
        return None
    
    def _find_array_hdf5(self, hdf5_file, keys):
        """Find array in HDF5 file by trying multiple keys"""
        for key in keys:
            if key in hdf5_file:
                arr = np.array(hdf5_file[key])
                if arr.size > 0:
                    return arr
        
        # If no specific key found, get the first large array
        for key in hdf5_file.keys():
            if not key.startswith('__'):
                arr = np.array(hdf5_file[key])
                if arr.size > 0 and arr.ndim >= 2:
                    return arr
        
        return None
    
    def _create_sample_data(self, dataset_name):
        """Create sample/demo data for testing"""
        sample_configs = {
            'Indian_pines': {
                'height': 145,
                'width': 145,
                'bands': 200,
                'classes': 16
            },
            'Pavia': {
                'height': 610,
                'width': 340,
                'bands': 102,
                'classes': 9
            },
            'PaviaU': {
                'height': 610,
                'width': 340,
                'bands': 103,
                'classes': 9
            },
            'Salinas': {
                'height': 512,
                'width': 217,
                'bands': 224,
                'classes': 16
            },
            'Chikusei': {
                'height': 2517,
                'width': 2335,
                'bands': 128,
                'classes': 15
            }
        }
        
        config = sample_configs.get(dataset_name, {'height': 145, 'width': 145, 'bands': 200, 'classes': 16})
        
        height = config['height']
        width = config['width']
        bands = config['bands']
        classes = config['classes']
        
        # Create realistic sample data
        self.data = np.zeros((height, width, bands), dtype=np.float32)
        
        for b in range(bands):
            x = np.linspace(0, 1, width)
            y = np.linspace(0, 1, height)
            X, Y = np.meshgrid(x, y)
            
            base = 0.5 + 0.3 * np.sin(b * np.pi / bands) * (X + Y)
            noise = np.random.randn(height, width) * 0.05
            self.data[:, :, b] = np.clip(base + noise, 0, 1)
        
        # Create ground truth with distinct regions
        self.gt = np.zeros((height, width), dtype=np.int32)
        
        regions_per_side = int(np.sqrt(classes))
        region_height = height // regions_per_side
        region_width = width // regions_per_side
        
        class_id = 1
        for i in range(regions_per_side):
            for j in range(regions_per_side):
                if class_id <= classes:
                    h_start = i * region_height
                    h_end = (i + 1) * region_height if i < regions_per_side - 1 else height
                    w_start = j * region_width
                    w_end = (j + 1) * region_width if j < regions_per_side - 1 else width
                    
                    self.gt[h_start:h_end, w_start:w_end] = class_id
                    class_id += 1
        
        unlabeled_mask = np.random.rand(height, width) < 0.1
        self.gt[unlabeled_mask] = 0
        
        self.data = np.ascontiguousarray(self.data)
        self.gt = np.ascontiguousarray(self.gt)