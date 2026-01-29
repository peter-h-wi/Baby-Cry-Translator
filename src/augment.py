import numpy as np
import librosa

def add_noise(data, noise_factor=0.005):
    """
    Add random white noise to the audio.
    """
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def time_stretch(data, rate=0.9):
    """
    Stretch the time (speed up or slow down/stretch).
    """
    return librosa.effects.time_stretch(y=data, rate=rate)

def pitch_shift(data, sampling_rate, n_steps=2):
    """
    Shift the pitch of the audio.
    """
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=n_steps)

def apply_augmentations(audio, sr):
    """
    Apply a random augmentation or a set of augmentations.
    Returns a list of augmented audio samples.
    """
    augmented_samples = []
    
    # 1. Noise
    augmented_samples.append(add_noise(audio))
    
    # 2. Time Stretch (slower)
    # augmented_samples.append(time_stretch(audio, rate=0.8))
    # Time stretch changes length, so we need to be careful if we enforce length elsewhere.
    # Our features extraction handles variable length somewhat by padding/truncating 
    # but let's stick to pitch shift which keeps duration relatively same (usually) 
    # or handle resize. Pad_or_truncate is called in preprocess, so we should allow it.
    
    # 3. Pitch Shift (higher)
    augmented_samples.append(pitch_shift(audio, sr, n_steps=2))
    
    # 4. Pitch Shift (lower)
    augmented_samples.append(pitch_shift(audio, sr, n_steps=-2))
    
    return augmented_samples
