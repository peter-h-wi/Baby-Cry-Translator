import librosa
import numpy as np
from . import config
from . import preprocess

def extract_features(audio):
    """
    Extract MFCC features from audio time series.
    Returns a 1D vector (mean of MFCCs over time).
    """
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC)
        # Take the mean of MFCCs over time to get a single vector per audio file
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def process_file(file_path):
    """
    Load, preprocess, and extract features from a file.
    """
    audio = preprocess.preprocess_audio(file_path)
    if audio is not None:
        return extract_features(audio)
    return None
