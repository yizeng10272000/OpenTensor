import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the style and resolution
sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

# Input and output paths
file_path = "exp/compare_result.csv"
output_dir = "compare_result"
os.makedirs(output_dir, exist_ok=True)

# Reading Data
df = pd.read_csv(file_path)
df["Use Projection"] = df["Use Projection"].astype(str)

# 1. Training time
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Use Projection", y="Train Time (sec)")
plt.title("Training Time vs Use of Projection")
plt.savefig(os.path.join(output_dir, "train_time_vs_projection.png"))
plt.close()

# 2. Inference time
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Use Projection", y="Infer Time (sec)")
plt.title("Inference Time vs Use of Projection")
plt.savefig(os.path.join(output_dir, "infer_time_vs_projection.png"))
plt.close()

# 3. Changes in inference time at each depth step
if "Depth" in df.columns and "Step Infer Time (sec)" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="Depth",
        y="Step Infer Time (sec)",
        hue="Use Projection",
        estimator="mean",
        errorbar="sd"
    )
    plt.title("Step Inference Time vs Depth")
    plt.savefig(os.path.join(output_dir, "step_infer_time_vs_depth.png"))
    plt.close()

# 4. Strategy distribution maximum value Pi Max changes
if "Step Pi Max" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="Depth",
        y="Step Pi Max",
        hue="Use Projection",
        estimator="mean",
        errorbar="sd"
    )
    plt.title("Step Pi Max vs Depth (Strategy Determinism)")
    plt.savefig(os.path.join(output_dir, "step_pi_max_vs_depth.png"))
    plt.close()

# 5. Final Train Loss
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Use Projection", y="Final Train Loss")
plt.title("Final Train Loss vs Use of Projection")
plt.savefig(os.path.join(output_dir, "final_train_loss_vs_projection.png"))
plt.close()

# 6. Peak Train Mem
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Use Projection", y="Peak Train Mem (MB)")
plt.title("Peak Training Memory vs Use of Projection")
plt.savefig(os.path.join(output_dir, "peak_train_mem_vs_projection.png"))
plt.close()

# 7. Total Model Params vs Final Loss
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Total Model Params", y="Final Train Loss", hue="Use Projection")
plt.title("Model Size vs Final Train Loss")
plt.savefig(os.path.join(output_dir, "model_size_vs_final_train_loss.png"))
plt.close()

print(f"Analysis completed, chart saved to directory: {output_dir}")
