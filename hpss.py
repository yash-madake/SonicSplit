# Harmonic-Percussive separation (Phase 3)

import librosa
import numpy as np

def separate_hpss(y, margin_harmonic=1.2, margin_percussive=1.2):
    """
    Decomposes an audio time series into harmonic and percussive components.
    Higher margins reduce bleed between the two tracks.
    """
    # Convert time-domain audio to STFT frequency matrix
    D = librosa.stft(y)
    
    # Perform mathematical separation using median filtering
    H, P = librosa.decompose.hpss(D, margin=(margin_harmonic, margin_percussive))
    
    # Invert the STFT matrices back into playable time-domain audio
    y_harmonic = librosa.istft(H)
    y_percussive = librosa.istft(P)
    
    return y_harmonic, y_percussive