import os
import time
import yaml
import csv
import re
import numpy as np

from codes.env import Environment
from codes.mcts import MCTS
from codes.net import Net
from codes.trainer import Trainer


def parse_infer_log(log_path):
    """Extract Q, score, depth, number of actions, etc. from infer_log.txt"""
    with open(log_path, 'r') as f:
        lines = f.readlines()

    Q_values, scores, depths, action_counts = [], [], [], []

    for i, line in enumerate(lines):
        if "Q:" in line:
            Q_values += list(map(float, re.findall(r'-?\d+\.\d+', line)))
        elif "scores:" in line:
            scores += list(map(float, re.findall(r'-?\d+\.\d+', line)))
        elif "depth=" in line:
            match = re.search(r"depth=(\d+)", line)
            if match:
                depths.append(int(match.group(1)))
        elif "Actions are:" in line:
            count = 0
            j = i + 1
            while j < len(lines) and '[' in lines[j]:
                count += 1
                j += 1
            action_counts.append(count)

    return {
        "mean_Q": np.mean(Q_values) if Q_values else "NA",
        "mean_score": np.mean(scores) if scores else "NA",
        "mean_depth": np.mean(depths) if depths else "NA",
        "mean_action_count": np.mean(action_counts) if action_counts else "NA"
    }


def run_opentensor(use_projection=True, projection_dim=None):
    with open("./config/S_4.yaml", "r") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)

    if use_projection:
        run_name = f"with_projection_dim{projection_dim}"
        kwargs["net"]["use_projection"] = True
        kwargs["net"]["projection_dim"] = projection_dim
    else:
        run_name = "without_projection"
        kwargs["net"]["use_projection"] = False
        kwargs["net"].pop("projection_dim", None)

    kwargs["trainer"]["exp_name"] = run_name

    data_path = "./data/100000_S%dT%d_scalar3_filtered.npy" % (
        kwargs["env"]["S_size"], kwargs["env"]["T"]
    )
    assert os.path.exists(data_path), f"Required data not found: {data_path}"

    net = Net(**kwargs["net"])
    mcts = MCTS(**kwargs["mcts"], init_state=None)
    env = Environment(**kwargs["env"], init_state=None)
    trainer = Trainer(**kwargs["trainer"], net=net, env=env, mcts=mcts, all_kwargs=kwargs)

    print(f"\n=== Training [{run_name}] ===")
    t0 = time.time()
    trainer.learn(resume=None, example_path=data_path, self_example_path=None)
    t1 = time.time()
    train_time = t1 - t0

    # Training-related indicators
    final_train_loss = getattr(trainer, "final_train_loss", "NA")
    train_steps = getattr(trainer, "train_steps", "NA")
    peak_train_mem = getattr(trainer, "peak_train_memory_MB", "NA")

    # Model parameter number
    total_params = sum(p.numel() for p in net.parameters())

    # Reasoning
    exp_base = f"./exp/{run_name}"
    all_subdirs = [os.path.join(exp_base, d) for d in os.listdir(exp_base) if os.path.isdir(os.path.join(exp_base, d))]
    if not all_subdirs:
        raise ValueError(f"❌ Cannot find subfolders in {exp_base}")
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    ckpt_path = os.path.join(latest_subdir, "ckpt", "latest.pth")
    assert os.path.exists(ckpt_path), f"❌ Checkpoint not found: {ckpt_path}"

    print(f"\n=== Inference [{run_name}] ===")
    t2 = time.time()
    steps = trainer.infer(resume=ckpt_path)
    t3 = time.time()
    infer_time = t3 - t2

    peak_infer_mem = getattr(trainer, "peak_infer_memory_MB", "NA")

    # === parse infer_log.txt ===
    infer_log_path = os.path.join(latest_subdir, "infer_log.txt")
    if os.path.exists(infer_log_path):
        infer_stats = parse_infer_log(infer_log_path)
    else:
        infer_stats = {
            "mean_Q": "NA",
            "mean_score": "NA",
            "mean_depth": "NA",
            "mean_action_count": "NA"
        }

    return {
        "run_name": run_name,
        "use_projection": use_projection,
        "projection_dim": projection_dim if projection_dim is not None else "NA",
        "train_time_sec": train_time,
        "infer_time_sec": infer_time,
        "mcts_steps": steps,
        "S_size": kwargs["env"]["S_size"],
        "T": kwargs["env"]["T"],
        "batch_size": kwargs["trainer"].get("batch_size", "NA"),
        "final_train_loss": final_train_loss,
        "train_steps": train_steps,
        "peak_train_mem_MB": peak_train_mem,
        "total_model_params": total_params,
        "peak_infer_mem_MB": peak_infer_mem,
        "mean_Q": infer_stats["mean_Q"],
        "mean_score": infer_stats["mean_score"],
        "mean_depth": infer_stats["mean_depth"],
        "mean_action_count": infer_stats["mean_action_count"]
    }


if __name__ == "__main__":
    print("Starting OpenTensor Compare Run\n")

    results = []

    # without projection
    result_no_proj = run_opentensor(use_projection=False)
    results.append(result_no_proj)

    # with projection 
    # for dim in range(16, 61, 4):
    #     res = run_opentensor(use_projection=True, projection_dim=dim)
    #     results.append(res)

    header = [
        "Run Name", "Use Projection", "Projection Dim",
        "Train Time (sec)", "Infer Time (sec)", "MCTS Steps",
        "S_size", "T", "Batch Size",
        "Final Train Loss", "Train Steps", "Peak Train Mem (MB)",
        "Total Model Params", "Peak Infer Mem (MB)",
        "Mean Q", "Mean Score", "Mean Depth", "Mean Action Count"
    ]

    print("\n=== Final Comparison Summary ===")
    print(" | ".join(f"{h:<18}" for h in header))
    print("-" * 180)
    for r in results:
        print(
            f"{str(r.get('run_name', 'NA')):<18} | {str(r.get('use_projection', 'NA')):<15} | {str(r.get('projection_dim', 'NA')):<14} | "
            f"{float(r.get('train_time_sec', 0)):<15.2f} | {float(r.get('infer_time_sec', 0)):<14.2f} | {str(r.get('mcts_steps', 'NA')):<10} | "
            f"{str(r.get('S_size', 'NA')):<6} | {str(r.get('T', 'NA')):<3} | {str(r.get('batch_size', 'NA')):<10} | "
            f"{str(r.get('final_train_loss', 'NA')):<16} | {str(r.get('train_steps', 'NA')):<11} | {str(r.get('peak_train_mem_MB', 'NA')):<17} | "
            f"{str(r.get('total_model_params', 'NA')):<18} | {str(r.get('peak_infer_mem_MB', 'NA')):<17} | "
            f"{str(r.get('mean_Q', 'NA')):<8} | {str(r.get('mean_score', 'NA')):<11} | {str(r.get('mean_depth', 'NA')):<11} | {str(r.get('mean_action_count', 'NA')):<16}"
        )

    os.makedirs("./exp", exist_ok=True)
    with open("./exp/compare_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "Run Name": r["run_name"],
                "Use Projection": r["use_projection"],
                "Projection Dim": r["projection_dim"],
                "Train Time (sec)": f"{r['train_time_sec']:.2f}",
                "Infer Time (sec)": f"{r['infer_time_sec']:.2f}",
                "MCTS Steps": r["mcts_steps"],
                "S_size": r["S_size"],
                "T": r["T"],
                "Batch Size": r["batch_size"],
                "Final Train Loss": r["final_train_loss"],
                "Train Steps": r["train_steps"],
                "Peak Train Mem (MB)": r["peak_train_mem_MB"],
                "Total Model Params": r["total_model_params"],
                "Peak Infer Mem (MB)": r["peak_infer_mem_MB"],
                "Mean Q": r["mean_Q"],
                "Mean Score": r["mean_score"],
                "Mean Depth": r["mean_depth"],
                "Mean Action Count": r["mean_action_count"]
            })

    print("\n✅ Comparison results saved to ./exp/compare_results.csv")
