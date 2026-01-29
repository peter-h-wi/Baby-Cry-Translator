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

## 📱 App-First Architecture (Baby Station vs Parent Station)

We simulate a "Smart Baby Monitor" system using two browser tabs or devices.

### 1. Baby Station (The Listener)
*   Place this device near the crib.
*   **Function**: Listens continuously, detects cries, and uploads events.
*   **Simulation**: Click "Simulate Cry Event" to generate a test alert.

### 2. Parent Station (The Receiver)
*   Parent carries this device.
*   **Function**: Receives push notifications (Auto-refresh) and allows Feedback.
*   **Feedback Loop**:
    *   **[✅]**: Confirm the prediction.
    *   **[❌]**: Correct the label (e.g., "Hungry" -> "Diaper"). This creates the "Golden Data" for Phase 2.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

### Run the App
```bash
streamlit run app.py
```
Open two browser tabs pointing to `http://localhost:8501`. Set one to **Baby Station** and the other to **Parent Station**.

### Train Models
```bash
# Baseline
python3 main.py

# Advanced (CNN)
python3 -m src.train_cnn
```
