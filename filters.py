# FIR vs IIR logic & phase response graphs (Phase 2)


import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import streamlit as st

def design_fir_bandpass(lowcut, highcut, fs, numtaps=1001, window='hamming'):
    """Designs a windowed FIR bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b = signal.firwin(numtaps, [low, high], pass_zero=False, window=window)
    a = [1.0] # FIR denominator is always 1
    return b, a

def design_iir_bandpass(lowcut, highcut, fs, order=4):
    """Designs a Butterworth IIR bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def plot_filter_comparisons(b_fir, a_fir, b_iir, a_iir, fs):
    """Plots Magnitude and Phase responses for both filters."""
    w_fir, h_fir = signal.freqz(b_fir, a_fir, worN=2000, fs=fs)
    w_iir, h_iir = signal.freqz(b_iir, a_iir, worN=2000, fs=fs)

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8))
    fig.patch.set_facecolor('#0e1117')

    # Magnitude Plot
    ax_mag.plot(w_fir, 20 * np.log10(np.abs(h_fir) + 1e-10), label='FIR (Windowed)', color='#1f77b4')
    ax_mag.plot(w_iir, 20 * np.log10(np.abs(h_iir) + 1e-10), label='IIR (Butterworth)', color='#ff7f0e', linestyle='--')
    ax_mag.set_title("Magnitude Response", color='white')
    ax_mag.set_ylabel("Amplitude (dB)", color='white')
    ax_mag.set_xlim([0, fs/2])
    ax_mag.set_ylim([-80, 5])
    ax_mag.grid(True, alpha=0.3)
    ax_mag.legend()
    ax_mag.tick_params(colors='white')

    # Phase Plot
    ax_phase.plot(w_fir, np.unwrap(np.angle(h_fir)), label='FIR Phase', color='#1f77b4')
    ax_phase.plot(w_iir, np.unwrap(np.angle(h_iir)), label='IIR Phase', color='#ff7f0e', linestyle='--')
    ax_phase.set_title("Phase Response", color='white')
    ax_phase.set_xlabel("Frequency (Hz)", color='white')
    ax_phase.set_ylabel("Phase (radians)", color='white')
    ax_phase.set_xlim([0, fs/2])
    ax_phase.grid(True, alpha=0.3)
    ax_phase.legend()
    ax_phase.tick_params(colors='white')
    
    # Styling
    for ax in [ax_mag, ax_phase]:
        ax.set_facecolor('#0e1117')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    plt.tight_layout()
    st.pyplot(fig)

def apply_filter(data, b, a):
    """Applies the filter using standard forward filtering (lfilter) to preserve phase characteristics."""
    return signal.lfilter(b, a, data)