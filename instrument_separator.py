import streamlit as st
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import io
import SonicSplit.visualizations as viz

INSTRUMENT_BANDS = {
    "Kick Drum": (20, 120),
    "Bass Guitar": (40, 250),
    "Acoustic Guitar": (80, 1200),
    "Male Vocals": (100, 800),
    "Sitar (Melody)": (150, 1200),
    "Female Vocals": (200, 1200),
    "Tabla (Dayan - Treble)": (200, 800),
    "Violin": (200, 3000),
    "Cymbals / Hi-Hats": (4000, 15000)
}

def isolate_instrument(y, sr, lowcut, highcut, numtaps=1001, window='hamming'):
    """Designs a linear-phase FIR filter and applies it to isolate an instrument."""
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # FIR Bandpass Filter Design
    b = signal.firwin(numtaps, [low, high], pass_zero=False, window=window)
    
    # Apply filter using FFT convolution for processing speed
    filtered_audio = signal.fftconvolve(y, b, mode='same')
    
    # Normalize audio to prevent clipping
    max_val = np.max(np.abs(filtered_audio))
    if max_val > 0:
        filtered_audio = filtered_audio / max_val
        
    return filtered_audio

def render_ui(y, sr):
    """Renders the UI and STFT graphs for the instrument separator."""
    st.subheader("🎧 Instrument Bandpass Separator")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_instrument = st.selectbox("Target Instrument", options=list(INSTRUMENT_BANDS.keys()))
        window_type = st.selectbox("Window Function", ["hamming", "hann", "blackman"])
    with col2:
        low, high = INSTRUMENT_BANDS[selected_instrument]
        cutoff_range = st.slider("Fine-tune Frequency Range (Hz)", min_value=20, max_value=15000, value=(low, high), step=10)
        fir_taps = st.slider("FIR Taps (N)", min_value=101, max_value=2001, value=1001, step=100)
    
    if st.button("Isolate Instrument", type="primary"):
        with st.spinner(f"Applying {window_type.capitalize()} windowed FIR filter..."):
            isolated_y = isolate_instrument(y, sr, cutoff_range[0], cutoff_range[1], numtaps=fir_taps, window=window_type)
            
            out_buffer = io.BytesIO()
            wavfile.write(out_buffer, sr, np.int16(isolated_y * 32767))
            
            st.success(f"Separation Complete: {selected_instrument}")
            st.audio(out_buffer, format="audio/wav")
            
            st.markdown(f"*Visual proof of the FIR filter isolating the {cutoff_range[0]}Hz - {cutoff_range[1]}Hz band.*")
            viz.plot_stft_spectrogram(isolated_y, sr, title=f"{selected_instrument} Isolated Spectrum")