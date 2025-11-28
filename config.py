import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

# Dataset configurations
DATASET_CONFIG = {
    'Indian_pines': {
        'data_file': 'Indian_pines.mat',
        'gt_file': 'Indian_pines_gt.mat',
        'data_key': 'indian_pines',
        'gt_key': 'indian_pines_gt',
        'num_classes': 16,
        'ignored_labels': [0]
    },
    'Pavia': {
        'data_file': 'Pavia.mat',
        'gt_file': 'Pavia_gt.mat',
        'data_key': 'pavia',
        'gt_key': 'pavia_gt',
        'num_classes': 9,
        'ignored_labels': [0]
    },
    'PaviaU': {
        'data_file': 'PaviaU.mat',
        'gt_file': 'PaviaU_gt.mat',
        'data_key': 'paviaU',
        'gt_key': 'paviaU_gt',
        'num_classes': 9,
        'ignored_labels': [0]
    },
    'Salinas': {
        'data_file': 'salinas_corrected.mat',
        'gt_file': 'salinas_gt.mat',
        'data_key': 'salinas_corrected',
        'gt_key': 'salinas_gt',
        'num_classes': 16,
        'ignored_labels': [0]
    },
    'Chikusei': {
    'data_file': 'HyperspecVNIR_Chikusei_20140729.mat',
    'gt_file': 'HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat',
    'data_key': 'chikusei',  # ✓ Correct
    'gt_key': 'GT',  # ✓ Correct
    'num_classes': 19,
    'ignored_labels': [0]
}
}


# Class names
CLASS_NAMES = {
    'Indian_pines': [
        'Background', 'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
        'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 
        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
        'Stone-Steel-Towers'
    ],
    'Pavia': [
        'Background', 'Water', 'Trees', 'Asphalt', 'Self-Blocking Bricks',
        'Bitumen', 'Tiles', 'Shadows', 'Meadows', 'Bare Soil'
    ],
    'PaviaU': [
        'Background', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
        'Painted metal sheets', 'Bare Soil', 'Bitumen',
        'Self-Blocking Bricks', 'Shadows'
    ],
    'Salinas': [
        'Background', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2',
        'Fallow', 'Fallow_rough_plow', 'Fallow_smooth', 'Stubble',
        'Celery', 'Grapes_untrained', 'Soil_vinyard_develop',
        'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
        'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
        'Vinyard_untrained', 'Vinyard_vertical_trellis'
    ]
}

# Model hyperparameters
PATCH_SIZE = 11
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
TEST_SPLIT = 0.9