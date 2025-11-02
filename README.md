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

<img width="953" height="650" alt="{ABED63A1-30F5-4FB6-9769-6C110600CC24}" src="https://github.com/user-attachments/assets/28c341a9-fe50-40e5-b8bd-9a490f0a11f9" />


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

