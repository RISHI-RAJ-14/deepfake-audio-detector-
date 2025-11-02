# 🎙️ Deepfake Audio Detection for KYC Authentication

This Streamlit web app detects **AI-generated (deepfake) audio** using a hybrid **CNN-LSTM** model, feature-based explainability, and advanced **forensic analysis** metrics.

---

## 🚀 Features

| Category | Description |
|-----------|--------------|
| 🎧 **Single Audio Detection** | Upload an audio clip and get real/fake prediction with confidence score |
| 🧠 **Explainability (Grad-CAM)** | Visualize which time-frequency regions influenced the model’s decision |
| 🔍 **Forensic Analysis** | Compute hand-crafted forensic scores — pitch jitter, spectral bursts, harmonicity, etc. |
| ⚖️ **Real vs Fake Comparison** | Compare real and fake audios side by side (waveforms, MFCCs, spectrograms) |

---

## 🧰 Project Structure

project/
│
├── app.py                       # 🎯 Main Streamlit entry file (handles routing + sidebar)
│
├── single_audio_page.py          # 🎧 Detect and explain deepfake for a single uploaded audio
├── compare_page.py               # ⚖️ Compare real vs fake audios side by side
├── advanced_features_page.py     # 🧠 Perform forensic and advanced acoustic analyses
│
├── cnn_lstm_deepfake_model.h5    # 🧩 Trained CNN-LSTM model (real vs fake classifier)
│
├── utils/                        # ⚙️ Core utility modules
│   ├── preprocessing.py          # 🔊 Audio loading, trimming, feature extraction (MFCCs, etc.)
│   ├── plotting.py               # 📊 Visualization helpers (waveform, spectrogram, MFCC plots)
│   ├── model_utils.py            # 🧠 Model loading, inference, and caching utilities
│   ├── explainability.py         # 🔥 Grad-CAM heatmaps and explainability visualizations
│   ├── advanced_features.py      # 🎵 Extracts advanced spectral and prosodic features
│   └── forensics.py              # 🔍 Forensic metrics (pitch jitter, harmonicity, fade mismatch)
│
├── requirements.txt              # 📦 Dependency list for Streamlit or local environment
├── README.md                     # 📘 Project documentation (overview, setup, usage)
└── screenshots/ (optional)       # 🖼️ Demo images for README or Streamlit Cloud preview

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/deepfake-audio-detector.git
   cd deepfake-audio-detector

2. **Create virtual environment (optional but recommended)**

python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate

3. **Install dependencies**

pip install -r requirements.txt

4. **Run the Streamlit app**

streamlit run app.py

