SonicSplit: Interactive Audio Separator

# InstruExtract: DSP Audio Separator

An interactive Python web application for isolating specific musical instruments from mixed audio and video files using Digital Signal Processing (DSP) techniques. Built with Streamlit and SciPy, this tool allows users to apply Linear Phase FIR filters to extract both Western and Indian Classical instruments based on their frequency bands.

## 🚀 Features
* **Multi-Format Media Support:** Ingests `.wav`, `.mp3`, and strips raw audio directly from video files (`.mp4`, `.mkv`) using Librosa and FFmpeg.
* **Culturally Diverse Presets:** 13 built-in frequency band presets covering Western instruments (Kick Drum, Vocals, Guitars) and Indian Classical instruments (Tabla, Sitar, Bansuri).
* **Interactive Filter Design:** Dynamically compare Rectangular, Hamming, Hanning, and Blackman windowing techniques and visualize the resulting Frequency Response curve.
* **Fast Convolution Processing:** Applies the filter using FFT-based circular convolution for rapid execution on large audio arrays.
* **Z-Plane Stability Proof:** Automatically generates a Pole-Zero constellation plot to mathematically prove Bounded-Input Bounded-Output (BIBO) stability.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **DSP Math Engine:** SciPy (`scipy.signal`), NumPy
* **Audio Extraction:** Librosa, FFmpeg
* **Visualization:** Matplotlib

## 📚 Academic DSP Concepts Applied
This project was developed to practically implement foundational DSP theory:
* **Module I (Causality & Stability):** Implements strictly causal FIR filters and evaluates stability using Z-transform Pole-Zero plots.
* **Module II (DFT & Linear Filtering):** Uses the Fast Fourier Transform (FFT) to generate magnitude spectrums and applies filters via Fast Circular Convolution in the frequency domain.
* **Module IV (FIR Filter Design):** Designs bandpass filters using difference equations and compares various windowing techniques (Hamming, Hann, Blackman) to minimize the Gibbs phenomenon.

## 💻 Installation & Usage

**1. Install System Dependencies**
You must have [FFmpeg](https://ffmpeg.org/download.html) installed on your system to extract audio from video files.
* Windows: `winget install ffmpeg`
* Mac: `brew install ffmpeg`
* Linux: `sudo apt install ffmpeg`

**2. Install Python Libraries**
```bash
pip install streamlit scipy numpy matplotlib librosa
