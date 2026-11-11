import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif, mutual_info_classif
import os

os.makedirs("results/figures", exist_ok=True)

# Configuration
CSV_PATH = "data/jem_features_dataset.csv"

FEATURE_COLS = [
    "Blade_Pass_Hz",
    "Shaft_Rate_Hz",
    "Est_Blade_Count_Int",
    "Doppler_BW_Hz",
    "Spectral_Entropy"
]

SHAFT_COL = "Shaft_Rate_Hz"
BPF_COL = "Blade_Pass_Hz"
CLASS_ID_COL = "Class_ID"

# Mapping from Class_ID (0-4) to report labels (A-E) and true blade counts
CLASS_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
CLASS_ORDER = ["A", "B", "C", "D", "E"]
TRUE_BLADE_COUNTS = {
    "A": 10,
    "B": 14,
    "C": 24,
    "D": 18,
    "E": 32
}

# Set to True to drop NaN rows, or False to impute with median + add a failure flag
DROP_NANS = False


def main():
    # Load Data
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} total rows from '{CSV_PATH}'.")

    # Verify feature columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in CSV: {missing}\nAvailable columns: {list(df.columns)}")

    # Ensure Class label column exists (use 'Class' if present, otherwise map 'Class_ID' 0-4 -> A-E)
    if "Class" in df.columns and df["Class"].isin(CLASS_ORDER).any():
        df["Class_Label"] = df["Class"]
    elif CLASS_ID_COL in df.columns:
        df["Class_Label"] = df[CLASS_ID_COL].map(CLASS_MAP)
    else:
        raise ValueError(f"Neither 'Class' nor '{CLASS_ID_COL}' found in CSV columns: {list(df.columns)}")

    # Handle NaN Rows
    df.loc[df[BPF_COL] == 0, BPF_COL] = np.nan

    nan_count = df[FEATURE_COLS].isna().any(axis=1).sum()
    if DROP_NANS:
        df_clean = df.dropna(subset=FEATURE_COLS + ["Class_Label"]).copy()
        df_clean["Extraction_Failed"] = 0
        active_features = FEATURE_COLS.copy()
        print(f"Dropped {nan_count} rows containing NaNs. Remaining samples: {len(df_clean)}")
    else:
        df_clean = df.dropna(subset=["Class_Label"]).copy()
        df_clean["Extraction_Failed"] = df_clean[FEATURE_COLS].isna().any(axis=1).astype(int)
        for col in FEATURE_COLS:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        active_features = FEATURE_COLS + ["Extraction_Failed"]
        print(f"Imputed {nan_count} NaN rows with medians and added 'Extraction_Failed' flag.")

    X = df_clean[active_features]
    y = df_clean["Class_Label"]

    # Correlation and Feature Selection
    print("\n" + "=" * 60)
    print("1A. FEATURE-TO-FEATURE CORRELATION MATRIX")
    print("=" * 60)
    corr_matrix = X.corr()
    print(corr_matrix.round(3).to_string())

    # Flag highly correlated pairs (|r| > 0.85)
    print("\n--- High Collinearity Check (|r| > 0.85) ---")
    found_high_corr = False
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            r_val = corr_matrix.iloc[i, j]
            if abs(r_val) > 0.85:
                print(f"  * {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: r = {r_val:.3f}")
                found_high_corr = True
    if not found_high_corr:
        print("  None found above 0.85.")

    print("\n" + "=" * 60)
    print("1B. CLASS RELEVANCE (ANOVA F-TEST & MUTUAL INFO)")
    print("=" * 60)
    f_scores, p_values = f_classif(X, y)
    mi_scores = mutual_info_classif(X, y, random_state=42)

    relevance_df = pd.DataFrame({
        "Feature": active_features,
        "ANOVA_F": f_scores,
        "p_value": p_values,
        "Mutual_Info": mi_scores
    }).sort_values(by="ANOVA_F", ascending=False)

    print(relevance_df.to_string(index=False))

    # Save tables to CSV
    corr_matrix.round(4).to_csv("results/output_correlation_matrix.csv")
    relevance_df.to_csv("results/output_feature_relevance.csv", index=False)

    # Generate Visualizations
    sns.set_theme(style="whitegrid")
    palette_colors = sns.color_palette("tab10", n_colors=len(CLASS_ORDER))
    color_map = dict(zip(CLASS_ORDER, palette_colors))

    # Plot 1: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Extracted JEM Feature Correlation Matrix", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig("results/figures/plot1_correlation_heatmap.png", dpi=300)
    plt.close()

    # Plot 2: Scatter of Blade-Pass vs. Shaft Rate with y = N*x Reference Lines
    df_valid = df_clean[df_clean["Extraction_Failed"] == 0]

    plt.figure(figsize=(10, 6.5))
    ax = sns.scatterplot(
        data=df_valid,
        x=SHAFT_COL,
        y=BPF_COL,
        hue="Class_Label",
        hue_order=CLASS_ORDER,
        palette=color_map,
        alpha=0.75,
        s=45
    )

    # Plot dashed reference lines y = N * x for each class
    x_max = df_valid[SHAFT_COL].max() * 1.05
    y_max = df_valid[BPF_COL].max() * 1.08
    x_vals = np.linspace(0, x_max, 200)

    for cls in CLASS_ORDER:
        n_blades = TRUE_BLADE_COUNTS[cls]
        plt.plot(
            x_vals,
            n_blades * x_vals,
            linestyle="--",
            linewidth=1.5,
            color=color_map[cls],
            alpha=0.85,
            label=f"Class {cls} theory (N={n_blades})"
        )

    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.title("Blade-Pass Frequency vs. Shaft Rate by Engine Class", fontsize=13, pad=12)
    plt.xlabel("Estimated Shaft Rate (Hz)")
    plt.ylabel("Estimated Blade-Pass Frequency (Hz)")
    plt.legend(title="Class & Theoretical Slope", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("results/figures/plot2_bpf_vs_shaft_scatter.png", dpi=300)
    plt.close()

    # Plot 3: Box Plots of Each Feature by Class (Ordered A-E)
    n_feats = len(FEATURE_COLS)
    n_cols = 3
    n_rows = int(np.ceil(n_feats / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 4.2 * n_rows))
    axes = np.array(axes).flatten()

    for idx, col in enumerate(FEATURE_COLS):
        sns.boxplot(
            data=df_clean,
            x="Class_Label",
            y=col,
            hue="Class_Label",
            order=CLASS_ORDER,
            hue_order=CLASS_ORDER,
            ax=axes[idx],
            palette=color_map,
            legend=False
        )
        axes[idx].set_title(f"{col} by Class", fontsize=11)
        axes[idx].set_xlabel("Engine Class")
        axes[idx].set_ylabel(col)

    # Remove unused subplots
    for j in range(n_feats, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("results/figures/plot3_feature_distributions_by_class.png", dpi=300)
    plt.close()

    print("\nDone. Saved 3 high-res plots and 2 summary CSVs in results/figures directory.")

if __name__ == "__main__":
    main()