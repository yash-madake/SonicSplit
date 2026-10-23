# 🎧 SonicSplit V2: Advanced DSP Audio Studio

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![SciPy & NumPy](https://img.shields.io/badge/DSP-SciPy%20%7C%20NumPy-00599C.svg)](https://scipy.org/)
[![Librosa](https://img.shields.io/badge/Audio-Librosa-yellow.svg)](https://librosa.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Author:** ""  
**Domain:** Digital Signal Processing (DSP) • Audio Source Separation • Discrete-Time Systems

---

## 📌 Executive Summary

**SonicSplit V2** is an interactive, modular Digital Signal Processing (DSP) toolkit designed to analyze, filter, decompose, and isolate audio signals without relying on opaque, resource-heavy deep learning models. 

By combining discrete-time system theory, **Fast Fourier Transform (FFT)** spectral energy scanning, **Short-Time Fourier Transform (STFT)** matrix decomposition, **Linear-Phase Finite Impulse Response (FIR)** windowing, and **$Z$-Transform inverse delay modeling**, SonicSplit demonstrates how classic DSP techniques solve audio engineering problems with mathematical transparency and real-time execution.

---

## 🚀 Core Modules & Features

### 1. Dynamic Time-Frequency Visualization (STFT)
- **Problem:** Static FFT magnitude plots only display averaged global frequency content without temporal context ($t$).
- **Solution:** Employs the Short-Time Fourier Transform (STFT) with logarithmic frequency scaling and dynamic decibel colormapping (`magma` colormap).
- **Mathematical Principle:**
  $$X(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - m] e^{-j\omega n}$$
  where $w[n]$ is the sliding window function and $m$ represents time shifts.
- **Insights:** Transients (drum strikes, percussive clicks) appear as vertical spectral bars; sustained tones (vocals, flutes, strings) appear as distinct horizontal harmonic tracks.

---

### 2. FIR vs. IIR Filter Benchmarking
- **Comparative Analysis:** Pitches linear-phase windowed FIR filters (`signal.firwin`) against recursive Butterworth Infinite Impulse Response (IIR) filters (`signal.butter`).
- **Mathematical & Phase Comparison:**
  - **FIR:** $H(z) = \sum_{n=0}^{N-1} b[n] z^{-n}$. All poles are strictly at the origin ($z=0$), guaranteeing unconditional Bounded-Input Bounded-Output (BIBO) stability and strict **linear phase** ($\theta(\omega) = -\alpha \omega$), preserving transient waveforms without group delay distortion.
  - **IIR:** $H(z) = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}}$. Achieves sharper cutoff transitions with fewer coefficients but introduces non-linear phase distortion.
- **Visual Validation:** Interactive side-by-side plots of magnitude ($|H(e^{j\omega})|_{\text{dB}}$) and unwrapped phase ($\angle H(e^{j\omega})$).

---

### 3. Harmonic-Percussive Source Separation (HPSS)
- **Problem:** Bandpass filters cannot separate instruments that share overlapping frequency bands (e.g., vocals and snare drums occupying 300–800 Hz).
- **Solution:** Implements median-filtering matrix decomposition directly across the STFT power spectrogram.
- **Mechanism:**
  - Sustained harmonics form smooth horizontal lines across time $\rightarrow$ extracted via median filtering along time axes.
  - Percussive transients form vertical broadband bursts $\rightarrow$ extracted via median filtering along frequency axes.
  - Generates independent harmonic (melody/vocals) and percussive (drums/beats) time-domain signals via inverse STFT (ISTFT).

---

### 4. Intelligent Spectral Energy Scanner & Instrument Separator
- **Spectral Energy Ratio Detection:** Computes the total signal energy $\sum |X[k]|$ and evaluates the energy ratio within discrete acoustic bands:
  $$\text{Ratio}_{\text{band}} = \frac{\sum_{k \in \text{band}} |X[k]|}{\sum_{\text{all } k} |X[k]|}$$
- **Dynamic UI:** Automatically identifies active acoustic ranges (e.g., Kick Drum, Bayan Tabla, Acoustic Guitar, Vocals, Sitar, Flute, Cymbals) and populates isolation presets.
- **Linear-Phase Bandpass Slicing:** Uses high-order symmetric windowed FIR convolution (`signal.fftconvolve`) to extract the selected instrument band without phase jitter.

---

### 5. Digital Echo & Reverb Cancellation ($Z$-Transform)
- **Acoustic Slapback Model:** A discrete-time single-echo signal is modeled as:
  $$y[n] = x[n] + \alpha x[n - D]$$
  where $\alpha$ is attenuation ($0 < \alpha < 1$) and $D = \tau \cdot f_s$ is the delay in samples.
- **System Transfer Function:**
  $$H(z) = 1 + \alpha z^{-D}$$
- **Inverse Filter:**
  $$H_{\text{inv}}(z) = \frac{1}{1 + \alpha z^{-D}}$$
- **Processing:** Cancels multipath acoustic delays via recursive inverse filtering, accompanied by interactive discrete impulse response plots.

---

## 📂 Project Architecture

```text
SonicSplit/
├── app.py                      # Master Streamlit UI & module router
├── visualizations.py           # Waveform & STFT log-spectrogram generation
├── filters.py                  # FIR vs. IIR design, frequency & phase response
├── hpss.py                     # Harmonic-Percussive separation via STFT
├── echo_cancel.py              # Z-transform inverse IIR echo cancellation
├── instrument_separator.py     # FFT energy detection & FIR bandpass isolation
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusions for audio files and caches
└── README.md                   # Comprehensive technical documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed.
- (Linux users) Ensure `libsndfile1` is available:
  ```bash
  sudo apt-get update && sudo apt-get install -y libsndfile1
  ```

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/SonicSplit.git
cd SonicSplit
```

### 2. Set Up a Virtual Environment
- **Windows (Command Prompt / PowerShell):**
  ```bash
  python -m venv .venv
  .\.venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📊 Presentation & Viva Defense Talking Points

When presenting this project to faculty or examiners, highlight these key engineering decisions:

1. **Why not just use an Equalizer (EQ)?**
   * *Talking Point:* A traditional EQ operates purely on static fixed frequency bands. SonicSplit couples FFT-based spectral energy ratio analysis with adaptive linear-phase FIR filtering to dynamically detect active instruments before isolation.
2. **Why does FIR matter over IIR in audio?**
   * *Talking Point:* FIR filters have strict linear phase ($\frac{d\theta}{d\omega} = \text{constant}$). This means all frequencies experience the exact same time delay (constant group delay), preserving musical transients like drum attacks. IIR filters produce phase dispersion, making transient signals sound smeared.
3. **How does HPSS surpass frequency band filtering?**
   * *Talking Point:* Vocals and snare drums overlap in the 500–1000 Hz region. A bandpass filter cannot separate them. HPSS separates them orthogonally in the STFT domain based on signal geometry (directional continuity) rather than frequency cutoffs.
4. **Stability Proof ($Z$-Plane):**
   * *Talking Point:* All causal FIR filters have their poles strictly at $z=0$ (the origin). Because the unit circle $|z| = 1$ is guaranteed to lie within the Region of Convergence (ROC), the system is inherently and unconditionally BIBO-stable.

---

## 📜 Dependencies
- **Streamlit**: Web application framework.
- **NumPy**: Matrix algebra and FFT computation.
- **SciPy**: Digital filter design (`firwin`, `butter`, `freqz`, `lfilter`, `fftconvolve`).
- **Matplotlib**: Spectrogram, waveform, and frequency/phase plotting.
- **Librosa**: Audio loading, Short-Time Fourier Transform (STFT), and HPSS decomposition.

---

## 👨‍💻 Author
****  
*Signal Processing & Audio Engineering Enthusiast*
