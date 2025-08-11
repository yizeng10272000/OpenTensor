import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.interpolate import interp1d
import numpy as np

# Setup
sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

# Paths
file_path = "exp/compare_results.csv"
output_dir = "compare_results"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv(file_path)
df["Use Projection"] = df["Use Projection"].astype(str)

# Add Projection Category
def label_projection(row):
    return "No_Proj" if row["Use Projection"] == "False" else f"Dim{int(row['Projection Dim'])}"

df["Projection Category"] = df.apply(label_projection, axis=1)

# === Function: Scatter plot with labels and interpolation line ===
def scatter_with_labels(data, x, y, label_col, hue, title, filename, interpolation=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = {"False": "#ff7f0e", "True": "#1f77b4"}

    # Draw scatter points
    sns.scatterplot(data=data, x=x, y=y, hue=hue, style=hue, s=100, palette=palette, ax=ax)

    # Add text labels
    x_offset = (data[x].max() - data[x].min()) * 0.03
    y_offset = (data[y].max() - data[y].min()) * 0.04

    for _, row in data.iterrows():
        label = "NoProj" if row["Use Projection"] == "False" else f"Dim{int(row['Projection Dim'])}"
        ax.text(row[x], row[y] + y_offset, label, fontsize=8, ha='center', va='bottom')

    # Add interpolation line
    if interpolation:
        # Drop NaNs and group by x to eliminate duplicates (take mean of y)
        cleaned_data = data[[x, y]].dropna().groupby(x).mean().reset_index().sort_values(by=x)

        x_vals = cleaned_data[x].values
        y_vals = cleaned_data[y].values

        # Only interpolate if enough points
        if len(x_vals) >= 2:
            interp_func = interp1d(x_vals, y_vals, kind='linear', fill_value="extrapolate")
            x_new = np.linspace(x_vals.min(), x_vals.max(), 300)
            y_new = interp_func(x_new)
            ax.plot(x_new, y_new, color='black', linestyle='--', label='Interpolation')

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(title=hue)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# === Function: Line plot ===
def line_plot(data, x, y, hue, title, filename):
    if x in data.columns and y in data.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x=x, y=y, hue=hue, estimator="mean", errorbar="sd")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

# === Function: Bar plot with numeric labels ===
def bar_plot_with_labels(data, x, y, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ensure consistent ordering
    ordered_cats = sorted(data[x].unique(), key=lambda v: (v != "No_Projection", v))
    palette = ["#1f77b4"] * len(ordered_cats)

    sns.barplot(data=data, x=x, y=y, hue=x, palette=palette, errorbar="sd", dodge=False, order=ordered_cats, ax=ax)
    ax.legend([], [], frameon=False)

    group_means = data.groupby(x)[y].mean()
    group_stds = data.groupby(x)[y].std()

    for i, category in enumerate(ordered_cats):
        mean_val = group_means.get(category, None)
        std_val = group_stds.get(category, 0)
        if mean_val is not None:
            ax.text(i, mean_val + std_val * 0.1, f"{mean_val:.3f}", ha='center', va='bottom', fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Projection Dimension Category")
    ax.set_ylabel(y)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# === 1. Model Params vs Final Train Loss ===
scatter_with_labels(
    df,
    x="Total Model Params",
    y="Final Train Loss",
    label_col="Projection Category",
    hue="Use Projection",
    title="Model Size vs Final Train Loss (with Projection Dim)",
    filename="model_size_vs_final_train_loss.png"
)

# === 2. Step Inference Time vs Depth ===
line_plot(
    data=df,
    x="Depth",
    y="Step Infer Time (sec)",
    hue="Use Projection",
    title="Step Inference Time vs Depth",
    filename="step_infer_time_vs_depth.png"
)

# === 3. Step Pi Max vs Depth ===
line_plot(
    data=df,
    x="Depth",
    y="Step Pi Max",
    hue="Use Projection",
    title="Step Pi Max vs Depth (Strategy Determinism)",
    filename="step_pi_max_vs_depth.png"
)

# === 4. Train Time vs Projection Dim ===
if "Train Time (sec)" in df.columns:
    bar_plot_with_labels(
        data=df,
        x="Projection Category",
        y="Train Time (sec)",
        title="Train Time vs Projection Dimension",
        filename="train_time_vs_projection_dim.png"
    )

# === 5. Inference Time vs Projection Dim ===
if "Infer Time (sec)" in df.columns:
    bar_plot_with_labels(
        data=df,
        x="Projection Category",
        y="Infer Time (sec)",
        title="Inference Time vs Projection Dimension",
        filename="infer_time_vs_projection_dim.png"
    )

print(f"✅ All analysis completed. Charts saved in: {output_dir}")
