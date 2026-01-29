# Baby Cry Translator (Acoustic Feature Base Model)

A Python-based machine learning model to translate baby cries into 5 categories: **Hungry, Tired, Belly Pain, Discomfort, Burping**.

## 🧠 Hybrid AI Strategy (The Roadmap)

We employ a **"Hybrid"** approach to handle the cold-start problem of data shortage.

### Phase 1: Launch (Current)
*   **Engine**: **Random Forest Classifier** (`src/train.py`)
*   **Why**: It is robust on small datasets (< 1000 samples) and handles class imbalance well using statistical features (MFCCs).
*   **Performance**: ~72% Accuracy (Stable).

### Phase 2: Evolution (Future)
*   **Engine**: **CNN (Convolutional Neural Network)** (`src/train_cnn.py`)
*   **Why**: As users correct predictions (Feedback Loop), data volume grows. Deep Learning (CNN) outperforms statistical models when data is abundant (> 10k samples) by seeing time-frequency patterns.
*   **Performance**: Currently ~36% (due to data shortage), but expected to surpass Phase 1 with more data.

---

## 🏗 System Architecture

1.  **Preprocessing**: Resample to 16kHz, Mono, Pad to 5s.
2.  **Data Augmentation**: Noise injection & Pitch shifting (Essential for minority classes).
3.  **Feature Engineering**:
    *   **Phase 1**: MFCC extraction (13 coeffs).
    *   **Phase 2**: Mel-Spectrogram generation (64 mels).
4.  **Inference**:
    *   The system can run on Edge devices (CCTV/Mobile) due to lightweight architecture.

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

### Train Baseline (Random Forest)
```bash
python3 main.py
```

### Train Advanced (CNN)
```bash
python3 -m src.train_cnn
```

## 🚀 Future Improvements

1.  **Real-Time Inference**: Connect microphone for live prediction.
2.  **Product Integration**: API for Mobile App/CCTV to send audio and receive feedback.
3.  **Continuous Learning**: Server pipeline to ingest feedback data and retrain the CNN nightly.
