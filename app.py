import streamlit as st
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import librosa
import tempfile
import os
import io

# Expand the page to fill the screen
st.set_page_config(page_title="DSP Separator", page_icon="🎧", layout="wide")

st.title("🎧 DSP Instrument Separator")
st.markdown("A frequency-band source separation tool built with Linear Phase FIR Filters.")

# Expanded Dictionary including Western and Indian instruments
INSTRUMENT_BANDS = {
    "Kick Drum": (20, 120),
    "Bass Guitar": (40, 250),
    "Tabla (Bayan - Bass)": (70, 250),
    "Acoustic Guitar": (80, 1200),
    "Male Vocals": (100, 800),
    "Sitar (Melody)": (150, 1200),
    "Female Vocals": (200, 1200),
    "Tabla (Dayan - Treble)": (200, 800),
    "Violin": (200, 3000),
    "Bansuri (Bamboo Flute)": (250, 2500),
    "Harmonium": (100, 3000),
    "Cymbals / Hi-Hats": (4000, 15000),
    "Custom (Manual Override)": (300, 3000)
}

# ==========================================
# UI: SIDEBAR (Controls)
# ==========================================
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # 1. File Uploader
    uploaded_file = st.file_uploader("1. Upload Media", type=["wav", "mp3", "mp4", "mkv"])
    
    st.write("---")
    
    # 2. Instrument Selection
    selected_instrument = st.selectbox("2. Target Instrument", options=list(INSTRUMENT_BANDS.keys()))
    default_low, default_high = INSTRUMENT_BANDS[selected_instrument]
    
    # 3. Fine-tune Slider
    st.write("3. Fine-tune Frequencies (Hz)")
    cutoff_range = st.slider(
        "Bandpass Range",
        min_value=20, max_value=15000,
        value=(default_low, default_high), step=10,
        label_visibility="collapsed"
    )
    
    st.write("---")
    
    # 4. Filter Options (Module IV comparison)
    window_type = st.selectbox("4. Window Function", ["hamming", "hann", "blackman", "boxcar"])
    
    st.write("---")
    apply_button = st.button("▶️ Run DSP Separation", type="primary", use_container_width=True)


# ==========================================
# MAIN APPLICATION LOGIC
# ==========================================
if uploaded_file is not None:
    
    # --- DATA LOADING ---
    file_extension = uploaded_file.name.split('.')[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        data, sample_rate = librosa.load(tmp_path, sr=None, mono=True)
    finally:
        os.remove(tmp_path)

    # --- DSP MATH SETUP ---
    nyquist = sample_rate / 2.0
    N = 1001 # Filter Taps
    
    # Phase 2: FFT Math
    chunk_size = min(len(data), sample_rate * 5)
    fft_spectrum = np.fft.rfft(data[:chunk_size])
    fft_freqs = np.fft.rfftfreq(chunk_size, 1.0 / sample_rate)
    magnitude_db = 20 * np.log10(np.abs(fft_spectrum) + 1e-10)

    # Phase 3: Filter Math
    fir_coeff = signal.firwin(
        numtaps=N, 
        cutoff=[cutoff_range[0], cutoff_range[1]], 
        window=window_type, 
        pass_zero=False, 
        fs=sample_rate
    )
    
    # Frequency Response Math
    w, h = signal.freqz(fir_coeff, [1.0], worN=2000, fs=sample_rate)
    mag_response_db = 20 * np.log10(np.abs(h) + 1e-10)

    # Phase 4: Z-Plane Math (Roots for Pole-Zero)
    zeros = np.roots(fir_coeff)
    poles = np.zeros(len(zeros))


    # ==========================================
    # UI: TABS (Main Display)
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["📊 Input Spectrum", "🎚️ Filter Separation", "📈 System Stability"])

    # --- TAB 1: INPUT ---
    with tab1:
        st.subheader("Original Audio")
        st.audio(uploaded_file)
        
        st.subheader("Frequency Magnitude Spectrum")
        fig_fft, ax_fft = plt.subplots(figsize=(10, 3))
        ax_fft.plot(fft_freqs, magnitude_db, color='#1f77b4', linewidth=0.5)
        ax_fft.set_xlim(20, 15000)
        ax_fft.set_ylabel("Magnitude (dB)")
        ax_fft.grid(True, alpha=0.3)
        st.pyplot(fig_fft)

    # --- TAB 2: FILTERING ---
    with tab2:
        # Beautiful metrics display
        m1, m2, m3 = st.columns(3)
        m1.metric("Filter Type", f"{window_type.capitalize()} FIR")
        m2.metric("Order (N)", f"{N} Taps")
        m3.metric("Passband", f"{cutoff_range[0]} - {cutoff_range[1]} Hz")
        
        st.subheader("Filter Frequency Response")
        fig_resp, ax_resp = plt.subplots(figsize=(10, 3))
        ax_resp.plot(w, mag_response_db, color='#ff7f0e', linewidth=1.5)
        ax_resp.axvline(cutoff_range[0], color='red', linestyle='--')
        ax_resp.axvline(cutoff_range[1], color='red', linestyle='--')
        ax_resp.set_xlim(0, 15000)
        ax_resp.set_ylim(-100, 5)
        ax_resp.set_ylabel("Magnitude (dB)")
        ax_resp.grid(True, alpha=0.3)
        st.pyplot(fig_resp)

        # Only run heavy convolution if the user clicks the red button in the sidebar
        if apply_button:
            with st.spinner("Executing Fast Circular Convolution..."):
                filtered_audio = signal.fftconvolve(data, fir_coeff, mode='same')
                
                max_val = np.max(np.abs(filtered_audio))
                if max_val > 0:
                    filtered_audio = filtered_audio / max_val
                
                filtered_audio_int16 = np.int16(filtered_audio * 32767)
                virtual_file = io.BytesIO()
                wavfile.write(virtual_file, sample_rate, filtered_audio_int16)
                
                st.success(f"Separation complete! Listening to: {selected_instrument}")
                st.audio(virtual_file, format="audio/wav")

    # --- TAB 3: STABILITY ---
    with tab3:
        with st.expander("📖 Mathematical Proof of Stability (Click to expand)", expanded=True):
            st.write(
                "An LTI system is stable if its Region of Convergence (ROC) includes the unit circle. "
                "Because this is a causal FIR filter, it has no feedback loop (the transfer function denominator is 1). "
                "Therefore, **all poles are located exactly at the origin ($z = 0$)**, strictly inside the unit circle. "
                "This mathematically guarantees Bounded-Input Bounded-Output (BIBO) stability."
            )
            
        fig_pz, ax_pz = plt.subplots(figsize=(5, 5))
        unit_circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', alpha=0.5)
        ax_pz.add_patch(unit_circle)
        ax_pz.axhline(0, color='black', lw=1, alpha=0.3)
        ax_pz.axvline(0, color='black', lw=1, alpha=0.3)
        
        ax_pz.scatter(np.real(zeros), np.imag(zeros), s=10, edgecolors='green', facecolors='none', marker='o', label='Zeros')
        ax_pz.scatter(np.real(poles), np.imag(poles), s=100, color='red', marker='x', label='Poles (at origin)')
        
        ax_pz.set_aspect('equal')
        ax_pz.set_xlim([-1.5, 1.5])
        ax_pz.set_ylim([-1.5, 1.5])
        ax_pz.legend(loc='upper right')
        ax_pz.grid(True, alpha=0.2)
        st.pyplot(fig_pz)

else:
    # Shown before any file is uploaded
    st.info("👈 Please upload a media file in the Sidebar to begin.")