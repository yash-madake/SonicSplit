# Z-transform delay/reverb removal (Phase 4)

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import streamlit as st

def remove_echo(data, sr, delay_sec, attenuation):
    """
    Applies an inverse IIR filter to cancel a specific delay-based echo.
    """
    delay_samples = int(delay_sec * sr)
    
    # Numerator for inverse filter: [1.0]
    b = np.array([1.0])
    
    # Denominator for inverse filter: 1 + alpha * z^-D
    a = np.zeros(delay_samples + 1)
    a[0] = 1.0
    a[-1] = attenuation
    
    # Apply the IIR filter
    cleaned_audio = signal.lfilter(b, a, data)
    return cleaned_audio

def plot_impulse_response(sr, delay_sec, attenuation):
    """Plots the impulse response of the echo cancellation filter."""
    delay_samples = int(delay_sec * sr)
    b = np.array([1.0])
    a = np.zeros(delay_samples + 1)
    a[0] = 1.0
    a[-1] = attenuation
    
    # Generate an impulse
    impulse = np.zeros(int(sr * 2)) # 2 seconds
    impulse[0] = 1.0
    
    # Get the impulse response of the IIR filter
    response = signal.lfilter(b, a, impulse)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    # Plot only the non-zero taps to make it readable
    taps = np.where(np.abs(response) > 0.001)[0]
    ax.vlines(taps / sr, 0, response[taps], color='#00ff00', linewidth=2)
    ax.plot(taps / sr, response[taps], 'o', color='#ff7f0e')
    
    ax.set_title("Filter Impulse Response (Echo Cancellation)", color='white')
    ax.set_xlabel("Time (seconds)", color='white')
    ax.set_ylabel("Amplitude", color='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)