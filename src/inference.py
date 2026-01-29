import joblib
import os
import numpy as np
from . import config
from . import features
from . import preprocess

# Define labels reverse mapping for display
ID_TO_LABEL = {v: k for k, v in config.LABELS.items()}

class CryPredictor:
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.model = None
        self.load_model()
        
    def load_model(self):
        if self.model_type == 'rf':
            model_path = os.path.join(config.BASE_DIR, 'model.pkl')
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print(f"Loaded Random Forest model from {model_path}")
            else:
                print(f"Model file not found: {model_path}")
        # Add 'cnn' implementation here later
        
    def predict(self, audio_path_or_array):
        if self.model is None:
            return None, 0.0
            
        # Extract features
        # If it's a path
        if isinstance(audio_path_or_array, str):
            feat = features.process_file(audio_path_or_array)
        else:
            # Assume it's a numpy array of raw audio? 
            # Needs to be preprocessed first.
            # But features.process_file does load+preprocess+extract.
            # features.extract_features takes raw audio.
            # Let's assume we pass raw audio array here if not string.
            # But we must ensure it's preprocessed (16k, mono, 5s).
            # This is tricky without knowing source SR. 
            # For simplicity, let's assume valid input or handle only paths for now 
            # or handle numpy array if it's already 16k mono.
             feat = features.extract_features(audio_path_or_array)

        if feat is None:
            return "Error", 0.0
            
        # Reshape for sklearn (1, n_features)
        feat = feat.reshape(1, -1)
        
        # Predict
        probs = self.model.predict_proba(feat)[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        label = ID_TO_LABEL.get(pred_idx, "Unknown")
        
        return label, confidence
