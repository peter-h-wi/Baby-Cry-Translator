import os

# Audio Configurations
SAMPLE_RATE = 16000
DURATION = 5  # seconds
HOP_LENGTH = 512
N_MFCC = 13
N_FFT = 2048

# Data Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Labels (Will be updated or used for mapping)
# Assuming the dataset has folder names that we can map. 
# We'll adapt this if the structure is different.
LABELS = {
    'hungry': 0,
    'tired': 1,
    'belly_pain': 2,
    'discomfort': 3,
    'burping': 4
}
