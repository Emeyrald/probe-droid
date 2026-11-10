import numpy as np

def preprocess_signal(signal):
    """
    Strips out massive zero-Doppler airframe features and normalizes to unit variance power.
    """
    signal_centered = signal - np.mean(signal)
    power = np.var(signal_centered)
    if power > 0:
        return signal_centered / np.sqrt(power)
    return signal_centered