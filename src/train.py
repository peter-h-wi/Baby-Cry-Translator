import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from . import config
from . import features
from . import preprocess

from . import augment

def load_dataset(data_path=config.DATA_RAW_DIR):
    """
    Load data and apply augmentation to balance classes.
    """
    X = []
    y = []
    file_paths = [] # Keep track for augmentation
    
    print(f"Scanning {data_path}...")
    
    # 1. First Pass: Load all original data
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                
                label = None
                for label_name, label_idx in config.LABELS.items():
                    if label_name in root.lower() or label_name in file.lower():
                        label = label_idx
                        break
                
                if label is not None:
                    feat = features.process_file(file_path)
                    if feat is not None:
                        X.append(feat)
                        y.append(label)
                        file_paths.append((file_path, label))
    
    if len(X) == 0:
        return np.array(X), np.array(y)

    # 2. Analyze Distribution
    X = np.array(X)
    y = np.array(y)
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print(f"Original Distribution: {class_counts}")
    
    max_count = max(class_counts.values())
    
    # 3. Augmentation Loop
    # Goal: Bring every class close to max_count (or at least boost them significantly)
    
    X_aug = list(X)
    y_aug = list(y)
    
    print("Applying Data Augmentation...")
    
    # We will iterate through loaded files and augment those belonging to minority classes
    for path, label in file_paths:
        # If this class is significantly smaller than the majority, augment it
        # Simple heuristic: if count < max_count * 0.5, we augment
        if class_counts[label] < max_count:
            # Calculate how many copies we aim for
            # We add 3 versions for every file (noise, pitch+, pitch-)
            # This might be too much for some, but let's try.
            
            raw_audio = preprocess.load_audio(path)
            if raw_audio is not None:
                # We need to pad/truncate before or after? 
                # augment functions take raw audio.
                # Pad first to ensure consistent length for time-based ops if needed, 
                # but augment usually handles raw. Let's pad after.
                
                # Apply augmentations
                augmented_audios = augment.apply_augmentations(raw_audio, config.SAMPLE_RATE)
                
                for aug_audio in augmented_audios:
                    # Preprocess (pad/truncate)
                    aug_audio = preprocess.pad_or_truncate(aug_audio)
                    # Extract features
                    feat = features.extract_features(aug_audio)
                    if feat is not None:
                        X_aug.append(feat)
                        y_aug.append(label)
    
    return np.array(X_aug), np.array(y_aug)

def train_model():
    X, y = load_dataset()
    
    if len(X) == 0:
        print("No data found! Please ensure data is in 'data/raw' and has generic folder names like 'hungry', 'tired/sleepy', 'diaper/pain'.")
        return

    print(f"Data loaded: {len(X)} samples.")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    # Using class_weight='balanced' to handle the imbalance (e.g. hungry has 380+ samples, others < 30)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Save
    model_path = os.path.join(config.BASE_DIR, 'model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Alignment Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model()
