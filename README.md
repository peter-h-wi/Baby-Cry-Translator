# Baby Cry Translator (Acoustic Feature Base Model)

A Python-based machine learning model to translate baby cries into 5 categories: **Hungry, Tired, Belly Pain, Discomfort, Burping**.

## 🧠 How It Works

The system follows a standard audio classification pipeline:

1.  **Data Loading (`src/train.py`)**
    *   Loads `.wav` files from `data/raw`.
    *   Automatically maps folder names to labels (e.g., `hungry` -> 0).

2.  **Preprocessing (`src/preprocess.py`)**
    *   **Resampling**: Converts all audio to **16,000 Hz** (standard for speech/human audio).
    *   **Mono Conversion**: Mixes stereo audio to a single channel.
    *   **Padding/Truncating**: forces every clip to be exactly **5.0 seconds** long.

3.  **Data Augmentation (`src/augment.py`)**
    *   *Solves the "Hungry Class Dominance" problem.*
    *   Identify minority classes (Tired, Belly Pain, etc.).
    *   Create new samples by:
        *   **Adding Noise**: Simulates background static.
        *   **Pitch Shifting**: Changes the pitch up/down by 2 steps.
    *   This balances the dataset so the model doesn't just guess "Hungry" every time.

4.  **Feature Extraction (`src/features.py`)**
    *   **MFCCs (Mel-Frequency Cepstral Coefficients)**: Extracts 13 coefficients that represent the "timbre" or "shape" of the sound.
    *   We average these over time to get a single vector of numbers for each file.

5.  **Modeling (`src/train.py`)**
    *   **Random Forest Classifier**: A robust algorithm that uses multiple decision trees.
    *   **Balanced Weights**: Penalizes mistakes on the minority classes more heavily.

## 🚀 Future Improvements (Roadmap)

### 1. Advanced Modeling (Deep Learning)
*   **CNN (Convolutional Neural Networks)**: Instead of averaging MFCCs, treat the MFCC spectrogram as an *image* and feed it into a CNN. This captures time-varying patterns (e.g., the *rhythm* of the cry).
*   **RNN/LSTM**: Good for sequential data, can learn the evolution of the cry over time.
*   **Transformer (AST)**: State-of-the-art for audio classification.

### 2. Real-Time Inference
*   Create a script (`predict.py`) that uses a microphone to record 5 seconds and predicts instantly.
*   Build a **Streamlit** or **Gradio** web app for an easy UI.

### 3. Data Expansion
*   The current dataset is very small for "Burping" and "Belly Pain".
*   Crowdsourcing or finding more datasets (e.g., from YouTube or Donate-A-Cry campaigns) is critical for reliability.

### 4. Mobile App
*   Convert the model to **TensorFlow Lite** or **CoreML** (for iOS) to run directly on a phone without internet.

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

```bash
python3 main.py
```
