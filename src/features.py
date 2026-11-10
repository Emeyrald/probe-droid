import numpy as np
from scipy.signal import find_peaks


def extract_jem_features(signal, fs):
    """
    Extracts noise-robust, leakage-free JEM features from a preprocessed time-domain signal.
    Signature is strictly (signal, fs) to prevent label leakage.
    """
    N = len(signal)
    df = fs / N  # Frequency resolution per bin (Hz)

    # Apply Hann window to suppress spectral leakage at high SNR
    window = np.hanning(N)
    windowed_sig = signal * window

    # Compute two-sided power spectrum and shift 0 Hz to center
    fft_vals = np.fft.fftshift(np.fft.fft(windowed_sig))
    power_spec = (np.abs(fft_vals) ** 2) / np.sum(window ** 2)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))

    # Mask out residual low-frequency airframe wobble (|f| <= 30 Hz) for cepstrum
    valid_mask = np.abs(freqs) > 30.0
    p_clean = np.where(valid_mask, power_spec, 0.0)

    # Isolate strictly positive frequencies (> 30 Hz) to eliminate the -bp to +bp 2x gap artifact
    pos_mask = freqs > 30.0
    f_pos = freqs[pos_mask]
    positive_bins = power_spec[pos_mask]

    # Dynamic Noise Floor (capped at 60 dB below peak so snr_db=None doesn't hit numerical floor)
    peak_power = np.max(positive_bins)
    noise_floor = max(float(np.median(positive_bins)), float(peak_power * 1e-6))

    # Smoothed Doppler Bandwidth Edge (prevents single noise spikes from inflating BW)
    kernel_size = max(5, int(250 / df))  # ~250 Hz moving average window
    kernel = np.ones(kernel_size) / kernel_size
    p_smooth = np.convolve(positive_bins, kernel, mode="same")

    active_bins = np.where(p_smooth > noise_floor * 5.0)[0]
    doppler_bw_hz = float(f_pos[active_bins[-1]]) if len(active_bins) > 0 else 0.0

    # Shaft-Rate Spacing via Power Cepstrum (computed first so it can cross-check blade-pass)
    log_spec = np.log10(np.maximum(p_clean, noise_floor))
    log_spec -= np.mean(log_spec)
    cepstrum = np.abs(np.fft.rfft(log_spec))
    quefrencies = np.fft.rfftfreq(N, d=df)  # Quefrency in seconds (1 / spacing_Hz)

    # Realistic shaft rotation rates: 30 Hz (1,800 RPM) to 300 Hz (18,000 RPM)
    q_mask = (quefrencies >= (1.0 / 300.0)) & (quefrencies <= (1.0 / 30.0))
    q_band = quefrencies[q_mask]
    c_band = cepstrum[q_mask]

    cep_med = np.median(c_band)
    cep_std = np.std(c_band)
    c_peaks, _ = find_peaks(c_band, height=cep_med + 6.0 * cep_std, prominence=4.5 * cep_std)

    if len(c_peaks) > 0:
        best_peak = c_peaks[np.argmax(c_band[c_peaks])]
        tallest_height = c_band[best_peak]
        shaft_rate_hz = float(1.0 / q_band[best_peak])

        # Check if 2x shaft rate (true period T) is also a strong peak (>= 65% of 2T peak)
        double_rate = 2.0 * shaft_rate_hz
        if double_rate <= 300.0:
            target_q = 1.0 / double_rate
            matching_peaks = [p for p in c_peaks if abs(q_band[p] - target_q) / target_q < 0.03]
            if matching_peaks:
                best_double = matching_peaks[np.argmax(c_band[matching_peaks])]
                if c_band[best_double] >= 0.65 * tallest_height:
                    shaft_rate_hz = float(1.0 / q_band[best_double])
    else:
        shaft_rate_hz = np.nan

    # Blade-Pass Spacing via Positive-Half Autocorrelation + 80% Pitch Rule
    p_strong_pos = np.maximum(0.0, positive_bins - noise_floor * 15.0)
    if np.sum(p_strong_pos) > 0:
        autocorr = np.correlate(p_strong_pos, p_strong_pos, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]  # Non-negative lags
        autocorr /= (autocorr[0] + 1e-12)

        # Search lags corresponding to realistic blade-pass frequencies (400 Hz to 6000 Hz)
        min_lag = max(1, int(400 / df))
        max_lag = min(len(autocorr) - 1, int(6000 / df))
        ac_slice = autocorr[min_lag:max_lag]
        ac_peaks, _ = find_peaks(ac_slice, prominence=0.05)

        if len(ac_peaks) > 0:
            peak_heights = ac_slice[ac_peaks]
            tallest = np.max(peak_heights)
            strong_candidates = ac_peaks[peak_heights >= 0.80 * tallest]
            first_lag = strong_candidates[0] + min_lag
            blade_pass_hz = float(first_lag * df)
        else:
            blade_pass_hz = 0.0
    else:
        blade_pass_hz = 0.0

    # Direct Spectral Check for Doubled Blade-Pass (bp / 2)
    # If blade_pass_hz is doubled (e.g., 2 * N * f_rot), a strong line will exist at bp / 2
    if blade_pass_hz >= 800.0:
        half_bp = blade_pass_hz / 2.0
        # Check if half_bp aligns with an integer multiple of shaft_rate_hz (if shaft rate is known)
        shaft_compatible = True
        if not np.isnan(shaft_rate_hz) and shaft_rate_hz > 0:
            half_ratio = half_bp / shaft_rate_hz
            shaft_compatible = (half_ratio >= 4.0) and (abs(half_ratio - round(half_ratio)) < 0.15)

        if shaft_compatible:
            # Look in a narrow window (±25 Hz) around half_bp in positive_bins
            half_band = positive_bins[np.abs(f_pos - half_bp) <= 25.0]
            if len(half_band) > 0 and np.max(half_band) >= 0.15 * peak_power:
                blade_pass_hz = half_bp

    # Final Integer-Ratio Cross-Check & Blade Count Calculation
    if not np.isnan(shaft_rate_hz) and shaft_rate_hz > 0 and blade_pass_hz > 0:
        current_ratio = blade_pass_hz / shaft_rate_hz
        if abs(current_ratio - round(current_ratio)) > 0.15:
            candidates = [0.5 * shaft_rate_hz, shaft_rate_hz, 2.0 * shaft_rate_hz]
            valid_cands = [c for c in candidates if 30.0 <= c <= 300.0]
            if valid_cands:
                shaft_rate_hz = float(
                    min(valid_cands, key=lambda c: abs((blade_pass_hz / c) - round(blade_pass_hz / c)))
                )

        est_blade_count_raw = float(blade_pass_hz / shaft_rate_hz)
        est_blade_count_int = float(np.round(est_blade_count_raw))
    else:
        est_blade_count_raw = np.nan
        est_blade_count_int = np.nan

    # Power Spectral Entropy (computed above noise floor)
    p_excess = np.maximum(0.0, positive_bins - noise_floor)
    if np.sum(p_excess) > 0:
        p_norm = p_excess / np.sum(p_excess)
        p_norm = p_norm[p_norm > 0]
        spectral_entropy = float(-np.sum(p_norm * np.log2(p_norm)))
    else:
        spectral_entropy = 0.0

    return {
        "Blade_Pass_Hz": blade_pass_hz,
        "Shaft_Rate_Hz": shaft_rate_hz,
        "Est_Blade_Count_Raw": est_blade_count_raw,
        "Est_Blade_Count_Int": est_blade_count_int,
        "Doppler_BW_Hz": doppler_bw_hz,
        "Spectral_Entropy": spectral_entropy,
    }
