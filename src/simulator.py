import numpy as np

def simulate_jem_signal(
    num_blades=12, 
    rpm=6000, 
    blade_length=0.4, 
    radar_freq=10e9, 
    fs=50000, 
    duration=0.3, 
    aspect_angle_deg=25.0, 
    airframe_rcs_db=20.0, 
    snr_db=20.0,
    num_scatterers=15,
    rng=None
):
    """
    Simulates Jet Engine Modulation (JEM) radar echo returns.
    
    Note on Physics Model Limitations:
    - Aspect angle is off the engine axis (0 = dead head-on). Because this model captures 
      Doppler phase shifts without blade-tilt amplitude modulation, aspect must stay > 0° 
      (e.g., 10°-60°) to generate non-zero blade variance.
    - SNR is defined strictly relative to the aggregated blade signal power, excluding the 
      static/low-frequency airframe return.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    if np.isclose(aspect_angle_deg, 0.0):
        raise ValueError("aspect_angle_deg cannot be 0.0 in a pure phase-Doppler model (yields zero blade variance). Keep within [10, 60] deg.")

    c = 3e8  
    lam = c / radar_freq  
    omega = 2 * np.pi * (rpm / 60.0)  
    t = np.arange(0, duration, 1/fs)
    theta_aspect = np.radians(aspect_angle_deg)

    # Per-blade amplitude and phase jitter exposes weak shaft-rate lines (f_rot)
    amp_jitter = rng.uniform(0.95, 1.05, num_blades)
    phase_jitter = rng.uniform(-0.02, 0.02, num_blades)

    jem_signal = np.zeros_like(t, dtype=complex)
    r_elements = np.linspace(0.1, blade_length, num_scatterers)

    for i in range(num_blades):
        phase_offset = i * (2 * np.pi / num_blades) + phase_jitter[i]
        for r in r_elements:
            r_t = r * np.sin(omega * t + phase_offset) * np.sin(theta_aspect)
            jem_signal += amp_jitter[i] * np.exp(1j * 4 * np.pi * r_t / lam)

    blade_power = np.var(jem_signal)
    
    # Airframe reflection with slight 5 Hz fluctuation so mean subtraction leaves realistic residual clutter
    airframe_linear_amp = np.sqrt(blade_power * (10**(airframe_rcs_db / 10.0)))
    airframe_fluctuation = 1.0 + 0.05 * np.sin(2 * np.pi * 5 * t)
    jem_signal += airframe_linear_amp * airframe_fluctuation

    # Complex White Gaussian Noise
    if snr_db is not None:
        snr_linear = 10**(snr_db / 10.0)
        noise_power = blade_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))
        jem_signal += noise

    return t, jem_signal
