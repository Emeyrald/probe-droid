import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.simulator import simulate_jem_signal
from src.preprocessing import preprocess_signal
from src.features import extract_jem_features
import numpy as np

def test_simulator_clean_physics():
    """Verifies exact line spacings, blade count ratio, and Doppler edge without noise."""
    rng = np.random.default_rng(42)
    num_blades = 10
    rpm = 3600  # 60 Hz shaft rate, 600 Hz blade-pass
    blade_length = 0.45
    aspect = 30.0
    fs = 50000

    expected_shaft = rpm / 60.0  # 60 Hz
    expected_blade_pass = num_blades * expected_shaft  # 600 Hz

    omega = 2 * np.pi * expected_shaft
    lam = 3e8 / 10e9
    expected_max_doppler = (2 * omega * blade_length * np.sin(np.radians(aspect))) / lam  # ~5655 Hz

    _, sig = simulate_jem_signal(
        num_blades=num_blades,
        rpm=rpm,
        blade_length=blade_length,
        fs=fs,
        duration=1.0,
        aspect_angle_deg=aspect,
        airframe_rcs_db=0.0,
        snr_db=None,
        rng=rng,
    )
    feats = extract_jem_features(preprocess_signal(sig), fs)

    assert np.isclose(feats["Blade_Pass_Hz"], expected_blade_pass, atol=5.0)
    assert np.isclose(feats["Shaft_Rate_Hz"], expected_shaft, atol=2.0)
    assert feats["Est_Blade_Count_Int"] == num_blades
    assert np.isclose(feats["Est_Blade_Count_Raw"], num_blades, atol=0.2)
    assert np.isclose(feats["Doppler_BW_Hz"], expected_max_doppler, atol=600.0)


def test_simulator_noisy_robustness():
    """Verifies spacing, blade count, and smoothed bandwidth robustness at 20 dB SNR."""
    rng = np.random.default_rng(101)
    num_blades = 14
    rpm = 9000  # 150 Hz shaft rate, 2100 Hz blade-pass
    blade_length = 0.30
    aspect = 25.0
    fs = 50000

    expected_shaft = rpm / 60.0  # 150 Hz
    expected_blade_pass = num_blades * expected_shaft  # 2100 Hz

    omega = 2 * np.pi * expected_shaft
    lam = 3e8 / 10e9
    expected_max_doppler = (2 * omega * blade_length * np.sin(np.radians(aspect))) / lam  # ~7966 Hz

    _, sig = simulate_jem_signal(
        num_blades=num_blades,
        rpm=rpm,
        blade_length=blade_length,
        fs=fs,
        duration=0.4,
        aspect_angle_deg=aspect,
        airframe_rcs_db=20.0,
        snr_db=20.0,
        rng=rng,
    )
    feats = extract_jem_features(preprocess_signal(sig), fs)

    assert np.isclose(feats["Blade_Pass_Hz"], expected_blade_pass, atol=15.0)
    assert np.isclose(feats["Shaft_Rate_Hz"], expected_shaft, atol=5.0)
    assert feats["Est_Blade_Count_Int"] == num_blades
    assert np.isclose(feats["Doppler_BW_Hz"], expected_max_doppler, atol=750.0)