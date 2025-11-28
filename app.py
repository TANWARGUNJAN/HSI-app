import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import os
import tempfile
import h5py
import seaborn as sns
from pathlib import Path
import shutil

from utils.data_loader import DataLoader
from utils.preprocessing import create_patches, split_dataset, apply_pca
from utils.visualization import plot_false_color, plot_ground_truth, plot_spectral_signature, plot_classification_map
from utils.metrics import calculate_metrics
from config import DATASET_CONFIG, CLASS_NAMES, PATCH_SIZE, UPLOADS_DIR

st.set_page_config(
    page_title="Hyperspectral Image Classification",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'gt' not in st.session_state:
    st.session_state.gt = None
if 'loader' not in st.session_state:
    st.session_state.loader = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = "Indian_pines"
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'uploaded_gt' not in st.session_state:
    st.session_state.uploaded_gt = None


# Replace your apply_theme() function with this updated version:

# Replace your apply_theme() function with this updated version:

def apply_theme():
    """Apply dark/light theme CSS"""
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        /* Main app background */
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        
        /* Sidebar dark theme - lighter shade */
        [data-testid="stSidebar"] {
            background-color: #1a1d29 !important;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            background-color: #1a1d29 !important;
        }
        
        /* Sidebar text color */
        [data-testid="stSidebar"] .element-container,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] span {
            color: #fafafa !important;
        }
        
        /* Radio buttons - lighter background */
        [data-testid="stSidebar"] .stRadio > div {
            background-color: #262b3d !important;
            border-radius: 10px;
            padding: 15px;
        }
        
        /* Select box - lighter background */
        [data-testid="stSidebar"] .stSelectbox > div > div {
            background-color: #262b3d !important;
            color: #fafafa !important;
            border-radius: 8px;
        }
        
        /* Text input - lighter background */
        [data-testid="stSidebar"] input {
            background-color: #262b3d !important;
            color: #fafafa !important;
            border: 1px solid #ff4b4b33 !important;
            border-radius: 8px;
        }
        
        /* Number input - lighter background */
        [data-testid="stSidebar"] input[type="number"] {
            background-color: #262b3d !important;
            color: #fafafa !important;
        }
        
        /* File uploader - lighter background */
        [data-testid="stSidebar"] [data-testid="stFileUploader"] {
            background-color: #262b3d !important;
            border-radius: 10px;
            padding: 15px;
        }
        
        [data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
            background-color: #1a1d29 !important;
            border: 2px dashed #ff4b4b !important;
            border-radius: 8px;
        }
        
        /* Buttons - lighter background */
        [data-testid="stSidebar"] button {
            background-color: #262b3d !important;
            color: #fafafa !important;
            border: 1px solid #ff4b4b33 !important;
            border-radius: 8px;
        }
        
        [data-testid="stSidebar"] button:hover {
            background-color: #ff4b4b !important;
            border-color: #ff4b4b !important;
            transform: translateY(-2px);
            transition: all 0.3s;
        }
        
        /* Divider */
        [data-testid="stSidebar"] hr {
            border-color: #ff4b4b33 !important;
        }
        
        /* Info boxes in sidebar - lighter background */
        [data-testid="stSidebar"] .stAlert {
            background-color: #262b3d !important;
            color: #fafafa !important;
        }
        
        /* Main tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #1e2130;
        }
        .stTabs [data-baseweb="tab"] {
            color: #fafafa;
        }
        
        /* Metrics - lighter cards for better visibility */
        .stMetric {
            background-color: #262b3d;
            padding: 15px;
            border-radius: 5px;
        }
        
        /* Metric label and value */
        .stMetric label {
            color: #a8b2d1 !important;
            font-weight: 500;
        }
        
        .stMetric [data-testid="stMetricValue"] {
            color: #fafafa !important;
            font-size: 2rem !important;
            font-weight: 600;
        }
        
        .stMetric [data-testid="stMetricDelta"] {
            color: #a8b2d1 !important;
        }
        
        /* Scrollbar for sidebar */
        [data-testid="stSidebar"]::-webkit-scrollbar {
            width: 8px;
        }
        
        [data-testid="stSidebar"]::-webkit-scrollbar-track {
            background: #0e1117;
        }
        
        [data-testid="stSidebar"]::-webkit-scrollbar-thumb {
            background: #262b3d;
            border-radius: 4px;
        }
        
        [data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
            background: #ff4b4b;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff;
            color: #262730;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #f0f2f6;
        }
        .stTabs [data-baseweb="tab"] {
            color: #262730;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

def load_dataset(dataset_name):
    """Load selected dataset"""
    with st.spinner(f"Loading {dataset_name}..."):
        loader = DataLoader(dataset_name)
        loader.load_data()
        
        st.session_state.loader = loader
        st.session_state.data = loader.data
        st.session_state.gt = loader.gt
        st.session_state.selected_dataset = dataset_name
        st.session_state.model = None
        st.session_state.predictions = None
        st.session_state.training_complete = False
        st.session_state.uploaded_data = None
        st.session_state.uploaded_gt = None
        
    st.success(f"✓ Loaded {dataset_name}")


def inspect_mat_keys(filepath):
    """Inspect a .mat file and return available keys"""
    keys = []
    try:
        # Try scipy
        mat_data = sio.loadmat(str(filepath))
        for key in mat_data.keys():
            if not key.startswith('__'):
                keys.append(key)
    except (NotImplementedError, OSError):
        # Try h5py
        try:
            with h5py.File(str(filepath), 'r') as f:
                def collect_keys(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        keys.append(name)
                f.visititems(collect_keys)
        except Exception:
            pass
    return keys


def load_uploaded_dataset(data_file, gt_file, data_key, gt_key):
    """Load dataset from uploaded files"""
    try:
        with st.spinner("Loading uploaded dataset..."):
            # Save uploaded files temporarily
            data_path = Path(UPLOADS_DIR) / data_file.name
            gt_path = Path(UPLOADS_DIR) / gt_file.name
            
            # Ensure uploads directory exists
            Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
            
            # Save files
            with open(data_path, "wb") as f:
                f.write(data_file.getbuffer())
            with open(gt_path, "wb") as f:
                f.write(gt_file.getbuffer())
            
            # Try loading with scipy first
            data = None
            gt = None
            
            try:
                mat_data = sio.loadmat(str(data_path))
                mat_gt = sio.loadmat(str(gt_path))
                
                if data_key not in mat_data:
                    available_keys = inspect_mat_keys(data_path)
                    raise KeyError(
                        f"Key '{data_key}' not found in {data_file.name}. "
                        f"Available keys: {available_keys}"
                    )
                
                if gt_key not in mat_gt:
                    available_keys = inspect_mat_keys(gt_path)
                    raise KeyError(
                        f"Key '{gt_key}' not found in {gt_file.name}. "
                        f"Available keys: {available_keys}"
                    )
                
                data = mat_data[data_key]
                gt = mat_gt[gt_key].squeeze()
                
            except (NotImplementedError, OSError):
                # Try HDF5
                try:
                    with h5py.File(str(data_path), 'r') as f:
                        if data_key not in f:
                            available_keys = list(f.keys())
                            raise KeyError(
                                f"Key '{data_key}' not found in {data_file.name}. "
                                f"Available keys: {available_keys}"
                            )
                        data = np.array(f[data_key])
                        # HDF5 might need transposing
                        if data.ndim == 3 and data.shape[0] < data.shape[2]:
                            data = np.transpose(data, (2, 1, 0))
                    
                    with h5py.File(str(gt_path), 'r') as f:
                        if gt_key not in f:
                            available_keys = list(f.keys())
                            raise KeyError(
                                f"Key '{gt_key}' not found in {gt_file.name}. "
                                f"Available keys: {available_keys}"
                            )
                        gt = np.array(f[gt_key]).squeeze()
                except Exception as e:
                    raise Exception(f"HDF5 loading failed: {e}")
            
            # Ensure correct shape
            if data.ndim == 2:
                data = data[:, :, np.newaxis]
            
            # FIX: Check if GT dimensions need transposing to match data
            if gt.shape[0] != data.shape[0] or gt.shape[1] != data.shape[1]:
                # Try transposing GT
                if gt.shape[0] == data.shape[1] and gt.shape[1] == data.shape[0]:
                    st.warning("⚠️ Ground truth dimensions were transposed to match data")
                    gt = gt.T
                else:
                    st.error(f"❌ Cannot fix dimension mismatch! Data: {data.shape[:2]}, GT: {gt.shape}")
                    return False
            
            # Verify dimensions match now
            if data.shape[0] != gt.shape[0] or data.shape[1] != gt.shape[1]:
                st.error(f"❌ Dimension mismatch! Data: {data.shape[:2]}, GT: {gt.shape}")
                st.error("Data and ground truth must have the same height and width.")
                return False
            
            # Store in session state
            st.session_state.data = data
            st.session_state.gt = gt
            st.session_state.uploaded_data = data
            st.session_state.uploaded_gt = gt
            st.session_state.selected_dataset = "Custom Upload"
            st.session_state.model = None
            st.session_state.predictions = None
            st.session_state.training_complete = False
            
            # Create a minimal loader object for compatibility
            class CustomLoader:
                def __init__(self, data, gt):
                    self.data = data
                    self.gt = gt
                    self.height = data.shape[0]
                    self.width = data.shape[1]
                    self.num_bands = data.shape[2]
            
            st.session_state.loader = CustomLoader(data, gt)
            
        st.success(f"✓ Loaded custom dataset: Data {data.shape}, GT {gt.shape}")
        return True
        
    except KeyError as e:
        st.error(f"❌ Key Error: {e}")
        st.info("💡 **Tip**: Check the available keys in the error message and update the 'Data key' and 'GT key' fields accordingly.")
        return False
    except Exception as e:
        st.error(f"❌ Error loading uploaded dataset: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False

def train_classifier():
    """Train a RandomForest classifier"""
    data = st.session_state.data
    gt = st.session_state.gt
    
    if data is None or gt is None:
        st.error("Please load a dataset first!")
        return
    
    with st.spinner("Training classifier..."):
        try:
            # Debug: Print shapes
            st.info(f"Data shape: {data.shape}, GT shape: {gt.shape}")
            
            # Ensure data and gt have matching spatial dimensions
            if data.shape[0] != gt.shape[0] or data.shape[1] != gt.shape[1]:
                st.error(f"❌ Dimension mismatch! Data: {data.shape[:2]}, GT: {gt.shape[:2]}")
                st.error("Data and ground truth must have the same height and width.")
                return
            
            # Flatten ground truth if needed
            if gt.ndim > 2:
                gt = gt.squeeze()
            
            # Reshape data to (height*width, bands)
            height, width, bands = data.shape
            data_reshaped = data.reshape(-1, bands)
            gt_flat = gt.flatten()
            
            # Get valid pixels (non-zero labels)
            valid_mask = gt_flat > 0
            
            # Extract valid samples using the flattened mask
            valid_pixels = data_reshaped[valid_mask]
            valid_labels = gt_flat[valid_mask]
            
            # Check if we have valid samples
            if len(valid_pixels) == 0:
                st.error("❌ No valid labeled pixels found in the dataset!")
                return
            
            st.info(f"Found {len(valid_pixels)} valid labeled pixels")
            
            # Prepare features and labels
            X = valid_pixels
            y = valid_labels
            
            # Adjust labels to start from 0
            y = y - y.min()
            
            # Split train/test
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train model
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_test, y_pred)
            
            # Store results
            st.session_state.model = model
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.training_complete = True
            
            # Generate full prediction map
            pred_map = np.zeros_like(gt, dtype=np.int32)
            all_pixels = data_reshaped  # Use the already reshaped data
            all_preds = model.predict(all_pixels)
            pred_map_flat = (all_preds + gt_flat.min())
            pred_map = pred_map_flat.reshape(gt.shape)
            pred_map[gt == 0] = 0  # Set background to 0
            st.session_state.predictions = pred_map
            
            st.success(f"✓ Training complete! Test accuracy: {acc*100:.2f}%")
            
        except Exception as e:
            st.error(f"❌ Training error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# ============================================================================
# MAIN APP UI
# ============================================================================

# Apply theme
apply_theme()

st.title("🛰️ Hyperspectral Image Classification")
st.markdown("### Interactive viewer and classifier for hyperspectral datasets")

# Replace your ENTIRE sidebar section (starting around line 421) with this:

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Dark mode toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Theme**")
    with col2:
        if st.button("🌓" if st.session_state.dark_mode else "☀️", key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    st.divider()
    
    # Dataset selection mode
    data_source = st.radio(
        "Data Source",
        ["📦 Built-in Datasets", "📤 Upload Custom Dataset"],
        label_visibility="collapsed"
    )
    
    if data_source == "📦 Built-in Datasets":
        dataset_name = st.selectbox(
            "Select Dataset",
            options=list(DATASET_CONFIG.keys()),
            index=list(DATASET_CONFIG.keys()).index(st.session_state.selected_dataset) if st.session_state.selected_dataset in DATASET_CONFIG else 0
        )
        
        if st.button("📁 Load Dataset", use_container_width=True):
            load_dataset(dataset_name)
        
        # ✅ ADD THIS SECTION FOR BUILT-IN DATASETS
        st.divider()
        st.subheader("🤖 Classification")
        if st.button("🚀 Train Classifier", use_container_width=True, key="train_builtin"):
            train_classifier()
    
    else:  # Upload custom dataset
        st.subheader("Upload Dataset")
        
        # Initialize session state for detected keys
        if 'detected_data_keys' not in st.session_state:
            st.session_state.detected_data_keys = []
        if 'detected_gt_keys' not in st.session_state:
            st.session_state.detected_gt_keys = []
        
        data_file = st.file_uploader(
            "Data file (.mat)",
            type=["mat"],
            help="Upload the hyperspectral data .mat file"
        )
        
        gt_file = st.file_uploader(
            "Ground truth file (.mat)",
            type=["mat"],
            help="Upload the ground truth labels .mat file"
        )
        
        # Auto-detect keys for both files
        col_detect1, col_detect2 = st.columns(2)
        
        with col_detect1:
            if data_file is not None:
                if st.button("🔍 Detect Data Keys", help="Auto-detect variable names in data file"):
                    with st.spinner("Inspecting data file..."):
                        temp_path = Path(UPLOADS_DIR) / data_file.name
                        Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
                        with open(temp_path, "wb") as f:
                            f.write(data_file.getbuffer())
                        
                        keys = inspect_mat_keys(temp_path)
                        st.session_state.detected_data_keys = keys
                        
                        if keys:
                            st.success(f"✓ Found: {', '.join(keys)}")
                        else:
                            st.warning("Could not detect keys")
        
        with col_detect2:
            if gt_file is not None:
                if st.button("🔍 Detect GT Keys", help="Auto-detect variable names in GT file"):
                    with st.spinner("Inspecting GT file..."):
                        temp_path = Path(UPLOADS_DIR) / gt_file.name
                        Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
                        with open(temp_path, "wb") as f:
                            f.write(gt_file.getbuffer())
                        
                        keys = inspect_mat_keys(temp_path)
                        st.session_state.detected_gt_keys = keys
                        
                        if keys:
                            st.success(f"✓ Found: {', '.join(keys)}")
                        else:
                            st.warning("Could not detect keys")
        
        # Show key selection - use dropdown if detected, otherwise text input
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.detected_data_keys:
                data_key = st.selectbox(
                    "Data key",
                    options=st.session_state.detected_data_keys,
                    help="Select the variable name for the data"
                )
            else:
                data_key = st.text_input(
                    "Data key",
                    value="data",
                    help="Variable name in the .mat file for the data"
                )
                st.caption("👆 Click 'Detect Data Keys' to auto-detect")
        
        with col2:
            if st.session_state.detected_gt_keys:
                gt_key = st.selectbox(
                    "GT key",
                    options=st.session_state.detected_gt_keys,
                    help="Select the variable name for ground truth"
                )
            else:
                gt_key = st.text_input(
                    "GT key",
                    value="gt",
                    help="Variable name in the .mat file for ground truth"
                )
                st.caption("👆 Click 'Detect GT Keys' to auto-detect")
        
        st.caption("💡 **Tip**: Click the 'Detect Keys' buttons above to automatically find available variable names")
        
        if st.button("📤 Load Uploaded Dataset", use_container_width=True, disabled=(data_file is None or gt_file is None)):
            load_uploaded_dataset(data_file, gt_file, data_key, gt_key)
        
        st.divider()
        
        # Training section for custom uploads
        st.subheader("🤖 Classification")
        if st.button("🚀 Train Classifier", use_container_width=True, key="train_custom"):
            train_classifier()
# Main content
if st.session_state.data is None:
    st.info("👈 Please select and load a dataset from the sidebar to begin.")
else:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Visualization", "📈 Spectral Analysis", "🎯 Classification", "📊 Metrics"])
    
    # Tab 1: Visualization
    with tab1:
        st.header("Dataset Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("False Color Composite")
            try:
                fig = plot_false_color(st.session_state.data)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error plotting false color: {e}")
        
        with col2:
            st.subheader("Ground Truth Map")
            try:
                class_names = CLASS_NAMES.get(st.session_state.selected_dataset, None)
                fig = plot_ground_truth(st.session_state.gt, class_names)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error plotting ground truth: {e}")
    
    # Tab 2: Spectral Analysis
    with tab2:
        st.header("Spectral Signature Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Select Pixel")
            
            loader = st.session_state.loader
            
            # Use GT dimensions for bounds checking
            max_x = min(loader.height, st.session_state.gt.shape[0]) - 1
            max_y = min(loader.width, st.session_state.gt.shape[1]) - 1
            
            pixel_x = st.number_input(
                "X coordinate",
                min_value=0,
                max_value=max_x,
                value=min(loader.height // 2, max_x)
            )
            pixel_y = st.number_input(
                "Y coordinate",
                min_value=0,
                max_value=max_y,
                value=min(loader.width // 2, max_y)
            )
            
            # Validate coordinates before accessing
            if (0 <= pixel_x < st.session_state.gt.shape[0] and 
                0 <= pixel_y < st.session_state.gt.shape[1]):
                gt_value = st.session_state.gt[pixel_x, pixel_y]
                st.write(f"**Ground truth class:** {gt_value}")
            else:
                st.warning(f"⚠️ Pixel ({pixel_x}, {pixel_y}) is out of GT bounds ({st.session_state.gt.shape})")
                gt_value = None
        
        with col2:
            st.subheader("Spectral Signature")
            try:
                # Also validate for data access
                if (0 <= pixel_x < st.session_state.data.shape[0] and 
                    0 <= pixel_y < st.session_state.data.shape[1]):
                    fig = plot_spectral_signature(st.session_state.data, (pixel_x, pixel_y))
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning(f"⚠️ Pixel ({pixel_x}, {pixel_y}) is out of data bounds ({st.session_state.data.shape[:2]})")
            except Exception as e:
                st.error(f"Error plotting spectral signature: {e}")
    
    # Tab 3: Classification
    with tab3:
        st.header("Classification Results")
        
        if not st.session_state.training_complete:
            st.info("👈 Train a classifier from the sidebar to see results.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Ground Truth")
                try:
                    class_names = CLASS_NAMES.get(st.session_state.selected_dataset, None)
                    fig = plot_ground_truth(st.session_state.gt, class_names)
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Error: {e}")
            
            with col2:
                st.subheader("Predictions")
                try:
                    fig = plot_classification_map(st.session_state.predictions, title="Classification Map")
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Tab 4: Metrics
    with tab4:
        st.header("Performance Metrics")
        
        if not st.session_state.training_complete:
            st.info("👈 Train a classifier from the sidebar to see metrics.")
        else:
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            
            # Overall metrics
            col1, col2, col3 = st.columns(3)
            
            acc = accuracy_score(y_test, y_pred)
            with col1:
                st.metric("Overall Accuracy", f"{acc*100:.2f}%")
            
            from sklearn.metrics import cohen_kappa_score
            kappa = cohen_kappa_score(y_test, y_pred)
            with col2:
                st.metric("Kappa Coefficient", f"{kappa:.4f}")
            
            cm = confusion_matrix(y_test, y_pred)
            aa = np.mean(np.diag(cm) / np.sum(cm, axis=1))
            with col3:
                st.metric("Average Accuracy", f"{aa*100:.2f}%")
            
            st.divider()
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            plt.close(fig)
            
            st.divider()
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"))

# Footer
st.divider()
st.markdown(
    """<div style='text-align: center; color: gray;'>
    Built with Streamlit | Hyperspectral Image Classification
    </div>""",
    unsafe_allow_html=True
)
