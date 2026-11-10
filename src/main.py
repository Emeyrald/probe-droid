import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from simulator import simulate_jem_signal
from preprocessing import preprocess_signal
from features import extract_jem_features

os.makedirs("results/figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    fs = 50000
    duration = 0.3
    SAMPLES_PER_CLASS = 200  # 5 classes * 200 = 1,000 total signals

    engine_classes = {
        "Class_A_Micro_Turbojet":  {"num_blades": 10, "rpm_range": (8500, 11500), "blade_length": 0.22},
        "Class_B_Medium_UAV":      {"num_blades": 14, "rpm_range": (6500, 9500),  "blade_length": 0.30},
        "Class_C_Bypass_Turbofan": {"num_blades": 24, "rpm_range": (4500, 7200),  "blade_length": 0.55},
        "Class_D_Fighter_Engine":  {"num_blades": 18, "rpm_range": (6000, 8800),  "blade_length": 0.42},
        "Class_E_Heavy_Transport": {"num_blades": 32, "rpm_range": (3200, 5200),  "blade_length": 0.70},
    }

    # Generate Phase-2 Representative Figures
    plot_classes = list(engine_classes.items())[:3]
    fig, axes = plt.subplots(len(plot_classes), 3, figsize=(16, 11))

    for idx, (class_name, params) in enumerate(plot_classes):
        mid_rpm = np.mean(params["rpm_range"])
        t, raw_sig = simulate_jem_signal(
            num_blades=params["num_blades"],
            rpm=mid_rpm,
            blade_length=params["blade_length"],
            fs=fs,
            duration=duration,
            aspect_angle_deg=25.0,
            airframe_rcs_db=20.0,
            snr_db=20.0,
            rng=rng,
        )

        axes[idx, 0].plot(t[:2500], raw_sig.real[:2500], color="tab:blue", alpha=0.8)
        axes[idx, 0].set_title(f"{class_name} - Raw Signal (First 50ms)")
        axes[idx, 0].set_xlabel("Time (s)")

        f_raw, t_raw, Sxx_raw = spectrogram(
            raw_sig, fs, nperseg=64, noverlap=48, detrend=False, return_onesided=False
        )
        im1 = axes[idx, 1].pcolormesh(
            t_raw,
            np.fft.fftshift(f_raw),
            10 * np.log10(np.fft.fftshift(Sxx_raw, axes=0) + 1e-10),
            shading="gouraud",
            cmap="viridis",
        )
        axes[idx, 1].set_title("Raw Spectrogram (dB)")
        fig.colorbar(im1, ax=axes[idx, 1])

        proc_sig = preprocess_signal(raw_sig)
        f_proc, t_proc, Sxx_proc = spectrogram(
            proc_sig, fs, nperseg=64, noverlap=48, detrend=False, return_onesided=False
        )
        im2 = axes[idx, 2].pcolormesh(
            t_proc,
            np.fft.fftshift(f_proc),
            10 * np.log10(np.fft.fftshift(Sxx_proc, axes=0) + 1e-10),
            shading="gouraud",
            cmap="magma",
        )
        axes[idx, 2].set_title("Processed Spectrogram (dB)")
        fig.colorbar(im2, ax=axes[idx, 2])

    plt.tight_layout()
    plt.savefig("results/figures/jem_spectrograms.png", dpi=300)
    plt.close()

    # Generate Full Randomized Dataset (1,000 samples)
    print(f"Generating full dataset ({SAMPLES_PER_CLASS} samples/class, {SAMPLES_PER_CLASS * len(engine_classes)} total)...")
    dataset_records = []
    processed_signals = []
    class_labels = []

    for class_idx, (class_name, params) in enumerate(engine_classes.items()):
        for _ in range(SAMPLES_PER_CLASS):
            rpm_sample = rng.uniform(*params["rpm_range"])
            aspect_sample = rng.uniform(15.0, 55.0)
            snr_sample = rng.uniform(10.0, 25.0)

            _, s_raw = simulate_jem_signal(
                num_blades=params["num_blades"],
                rpm=rpm_sample,
                blade_length=params["blade_length"],
                fs=fs,
                duration=duration,
                aspect_angle_deg=aspect_sample,
                airframe_rcs_db=20.0,
                snr_db=snr_sample,
                rng=rng,
            )

            s_proc = preprocess_signal(s_raw)
            feats = extract_jem_features(s_proc, fs)

            processed_signals.append(s_proc.astype(np.complex64))
            class_labels.append(class_idx)

            record = {
                "Class": class_name,
                "Class_ID": class_idx,
                "True_Blades": params["num_blades"],
                "True_RPM": rpm_sample,
                "Aspect_Deg": aspect_sample,
                "SNR_dB": snr_sample,
                **feats,
            }
            dataset_records.append(record)

    df_dataset = pd.DataFrame(dataset_records)
    df_dataset.to_csv("data/jem_features_dataset.csv", index=False)

    np.savez_compressed(
        "data/jem_signals_dataset.npz",
        signals=np.array(processed_signals),
        labels=np.array(class_labels),
        class_names=np.array(list(engine_classes.keys())),
        fs=fs,
    )
    print("Saved tabular features to data/jem_features_dataset.csv")
    print("Saved preprocessed signals to data/jem_signals_dataset.npz\n")

    # Print Summary Statistics Across the Dataset
    feature_cols = [
        "Blade_Pass_Hz",
        "Shaft_Rate_Hz",
        "Est_Blade_Count_Int",
        "Doppler_BW_Hz",
        "Spectral_Entropy",
    ]
    summary_table = df_dataset.groupby("Class")[feature_cols].agg(["mean", "std", "min", "max"])

    print("--- FEATURE SUMMARY TABLE ACROSS DATASET ---")
    print(summary_table.round(2).to_string())
