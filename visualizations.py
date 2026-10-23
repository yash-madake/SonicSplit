# STFT Spectrograms & waveform plotting (Phase 1)

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import streamlit as st

def plot_waveform(y, sr, title="Audio Waveform"):
    """Generates a clean, modern waveform plot."""
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#1f77b4', alpha=0.8)
    ax.set_title(title, color='white')
    ax.set_ylabel("Amplitude")
    
    # Styling for a sleek dark mode look
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    
    st.pyplot(fig)

def plot_stft_spectrogram(y, sr, title="STFT Spectrogram (Log Frequency)"):
    """Generates a high-quality spectrogram using Short-Time Fourier Transform."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Calculate STFT
    D = librosa.stft(y)
    # Convert amplitude to Decibels (dB)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Plot using a visually striking colormap (magma)
    img = librosa.display.specshow(
        S_db, 
        x_axis='time', 
        y_axis='log', 
        sr=sr, 
        cmap='magma', 
        ax=ax
    )
    
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title, color='white')
    
    # Dark mode styling
    fig.patch.set_facecolor('#0e1117')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    
    st.pyplot(fig)