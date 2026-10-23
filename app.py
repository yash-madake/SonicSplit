import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import io
import scipy.io.wavfile as wavfile

# Custom DSP Modules
import visualizations as viz
import filters as flt
import hpss
import echo_cancel as ec
import instrument_separator as inst_sep

st.set_page_config(page_title="Advanced DSP Studio", page_icon="🔬", layout="wide")

st.title("🔬 Advanced DSP Audio Studio")
st.markdown("A modular toolkit for signal analysis, filtering, and source separation.")

# Sidebar Navigation & Upload
with st.sidebar:
    st.header("🎛️ Navigation")
    app_mode = st.selectbox(
        "Choose DSP Module",
        ["1. Signal Analysis (STFT)", 
         "2. FIR vs IIR Filtering", 
         "3. Harmonic-Percussive Separation", 
         "4. Echo & Reverb Cancellation",
         "5. Instrument Bandpass Separator"]
    )
    
    st.write("---")
    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

# Main Application Logic
if uploaded_file is not None:
    # Safely load the file via a temporary path
    file_extension = uploaded_file.name.split('.')[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with st.spinner("Loading audio data..."):
        # Load a 30-second snippet to keep processing fast during development
        y, sr = librosa.load(tmp_path, sr=None, mono=True, duration=30.0)
    os.remove(tmp_path)

    # Route to the correct module based on sidebar selection
    if app_mode == "1. Signal Analysis (STFT)":
        st.subheader("Time-Domain Analysis")
        st.audio(uploaded_file)
        viz.plot_waveform(y, sr)
        
        st.subheader("Time-Frequency Domain Analysis")
        st.markdown("*Notice how the STFT reveals both the frequency content and **when** those frequencies occur.*")
        viz.plot_stft_spectrogram(y, sr)
        
    elif app_mode == "2. FIR vs IIR Filtering":
        st.subheader("Filter Design & Analysis")
        st.markdown("Compare the linear phase of FIR filters against the computational efficiency of IIR filters.")
        
        col1, col2 = st.columns(2)
        with col1:
            cutoff_low = st.number_input("Low Cutoff (Hz)", min_value=20, max_value=10000, value=300)
            fir_taps = st.slider("FIR Taps (N)", min_value=101, max_value=2001, value=1001, step=100)
        with col2:
            cutoff_high = st.number_input("High Cutoff (Hz)", min_value=100, max_value=20000, value=3000)
            iir_order = st.slider("IIR Butterworth Order", min_value=2, max_value=10, value=4)

        # Generate filter coefficients
        b_fir, a_fir = flt.design_fir_bandpass(cutoff_low, cutoff_high, sr, numtaps=fir_taps)
        b_iir, a_iir = flt.design_iir_bandpass(cutoff_low, cutoff_high, sr, order=iir_order)

        # Plot Mathematical Responses
        flt.plot_filter_comparisons(b_fir, a_fir, b_iir, a_iir, sr)
        
        # Audio Processing
        if st.button("Apply Filters & Compare Audio", type="primary"):
            with st.spinner("Processing audio through both filters..."):
                filtered_fir = flt.apply_filter(y, b_fir, a_fir)
                filtered_iir = flt.apply_filter(y, b_iir, a_iir)
                
                # Normalize audio
                filtered_fir = filtered_fir / np.max(np.abs(filtered_fir))
                filtered_iir = filtered_iir / np.max(np.abs(filtered_iir))
                
                fir_buffer = io.BytesIO()
                iir_buffer = io.BytesIO()
                wavfile.write(fir_buffer, sr, np.int16(filtered_fir * 32767))
                wavfile.write(iir_buffer, sr, np.int16(filtered_iir * 32767))
                
                ac1, ac2 = st.columns(2)
                with ac1:
                    st.success("FIR Filtered Output")
                    st.audio(fir_buffer, format="audio/wav")
                with ac2:
                    st.warning("IIR Filtered Output")
                    st.audio(iir_buffer, format="audio/wav")
                    
    elif app_mode == "3. Harmonic-Percussive Separation":
        st.subheader("Harmonic-Percussive Source Separation (HPSS)")
        st.markdown("Isolate sustained instruments (vocals, strings) from transient strikes (drums, tabla) using STFT matrix decomposition.")
        
        col1, col2 = st.columns(2)
        with col1:
            h_margin = st.slider("Harmonic Margin (Strictness)", min_value=1.0, max_value=5.0, value=1.2, step=0.1)
        with col2:
            p_margin = st.slider("Percussive Margin (Strictness)", min_value=1.0, max_value=5.0, value=1.2, step=0.1)
            
        if st.button("Run HPSS Separation", type="primary"):
            with st.spinner("Decomposing audio matrices..."):
                y_harm, y_perc = hpss.separate_hpss(y, margin_harmonic=h_margin, margin_percussive=p_margin)
                
                # Normalize audio arrays to prevent clipping distortion
                y_harm = y_harm / np.max(np.abs(y_harm))
                y_perc = y_perc / np.max(np.abs(y_perc))
                
                h_buffer = io.BytesIO()
                p_buffer = io.BytesIO()
                wavfile.write(h_buffer, sr, np.int16(y_harm * 32767))
                wavfile.write(p_buffer, sr, np.int16(y_perc * 32767))
                
                ac1, ac2 = st.columns(2)
                with ac1:
                    st.success("Harmonic Output (Vocals/Melody)")
                    st.audio(h_buffer, format="audio/wav")
                with ac2:
                    st.info("Percussive Output (Drums/Beats)")
                    st.audio(p_buffer, format="audio/wav")
                    
    elif app_mode == "4. Echo & Reverb Cancellation":
        st.subheader("Digital Echo Cancellation (Z-Transform)")
        st.markdown(r"Removes delay-based echoes using the inverse transfer function $H_{inv}(z) = \frac{1}{1 + \alpha z^{-D}}$")
        
        col1, col2 = st.columns(2)
        with col1:
            delay = st.slider("Target Echo Delay (Seconds)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        with col2:
            alpha = st.slider("Echo Attenuation (Alpha)", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
            
        ec.plot_impulse_response(sr, delay, alpha)
        
        if st.button("Run Echo Cancellation", type="primary"):
            with st.spinner("Applying inverse IIR filter..."):
                cleaned_y = ec.remove_echo(y, sr, delay_sec=delay, attenuation=alpha)
                
                # Normalize to prevent clipping
                cleaned_y = cleaned_y / np.max(np.abs(cleaned_y))
                
                out_buffer = io.BytesIO()
                wavfile.write(out_buffer, sr, np.int16(cleaned_y * 32767))
                
                st.success("Echo Cancellation Complete")
                st.audio(out_buffer, format="audio/wav")
                
    elif app_mode == "5. Instrument Bandpass Separator":
        inst_sep.render_ui(y, sr)

else:
    st.info("👈 Please upload an audio file in the sidebar to begin analyzing signals.")