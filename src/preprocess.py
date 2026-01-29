import librosa
import numpy as np
from . import config

def load_audio(file_path):
    """
    Load audio file, resample to target sample rate, and convert to mono.
    """
    try:
        # librosa.load resamples to sr and converts to mono by default (mono=True)
        audio, _ = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def pad_or_truncate(audio):
    """
    Ensure the audio is exactly the target duration.
    """
    target_length = int(config.SAMPLE_RATE * config.DURATION)
    
    if len(audio) > target_length:
        return audio[:target_length]
    else:
        padding = target_length - len(audio)
        return np.pad(audio, (0, padding), mode='constant')

def preprocess_audio(file_path):
    """
    Full preprocessing pipeline for a single file.
    """
    audio = load_audio(file_path)
    if audio is not None:
        audio = pad_or_truncate(audio)
    return audio
