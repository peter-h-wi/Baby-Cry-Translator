import torch
from torch.utils.data import Dataset
import torchaudio
import os
import numpy as np
from . import config
from . import preprocess
from . import augment

class BabyCryDataset(Dataset):
    def __init__(self, X_paths, y_labels, transform=None, augment=False):
        """
        Args:
            X_paths: List of file paths.
            y_labels: List of integer labels.
            transform: Transformations (e.g. Spectrogram).
            augment: Boolean, whether to apply augmentation (Noise, Pitch Shift).
        """
        self.X_paths = X_paths
        self.y_labels = y_labels
        self.augment = augment
        # Spectrogram Transform
        # We define it here to be consistent
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=64
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.X_paths)

    def __getitem__(self, idx):
        path = self.X_paths[idx]
        label = self.y_labels[idx]
        
        # Load audio (we use our preprocess to ensure uniformSR/Mono/Length)
        # Preprocess returns numpy array, convert to tensor
        audio = preprocess.preprocess_audio(path)
        
        if audio is None:
            # Handle bad file - return a zero tensor
            # Ideally filter these out before creating dataset
            audio = np.zeros(int(config.SAMPLE_RATE * config.DURATION))
            
        # Data Augmentation (On-the-fly)
        if self.augment:
             # Randomly choose an augmentation
             choice = np.random.choice(['none', 'noise', 'pitch_up', 'pitch_down'])
             if choice == 'noise':
                 audio = augment.add_noise(audio)
             elif choice == 'pitch_up':
                 audio = augment.pitch_shift(audio, config.SAMPLE_RATE, n_steps=2)
             elif choice == 'pitch_down':
                 audio = augment.pitch_shift(audio, config.SAMPLE_RATE, n_steps=-2)
        
        # Convert to Tensor [1, Time]
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        
        # Generate Spectrogram [1, n_mels, Time]
        # Spectrogram shape: [1, 64, ~157] for 5s audio at 16k SR with 512 hop
        spec = self.mel_spectrogram(audio_tensor)
        spec = self.amplitude_to_db(spec)
        
        # Normalize (optional, but good for CNN)
        # spec = (spec - spec.mean()) / spec.std()
        
        return spec, label
