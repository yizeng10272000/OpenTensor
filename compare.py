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
    trainer.infer(resume=ckpt_path)
    t3 = time.time()
    infer_time = t3 - t2

    peak_infer_mem = getattr(trainer, "peak_infer_memory_MB", "NA")

    return {
        "run_name": run_name,
        "use_projection": use_projection,
        "projection_dim": projection_dim if projection_dim is not None else "NA",
        "train_time_sec": train_time,
        "infer_time_sec": infer_time,
        "final_train_loss": final_train_loss,
        "peak_train_mem_MB": peak_train_mem,
        "total_model_params": total_params,
        "peak_infer_mem_MB": peak_infer_mem,
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
        "Train Time (sec)", "Infer Time (sec)",
        "Final Train Loss", "Peak Train Mem (MB)",
        "Total Model Params", "Peak Infer Mem (MB)"
    ]

    print("\n=== Final Comparison Summary ===")
    print(" | ".join(f"{h:<20}" for h in header))
    print("-" * 180)
    for r in results:
        print(
            f"{str(r['run_name']):<20} | {str(r['use_projection']):<15} | {str(r['projection_dim']):<14} | "
            f"{r['train_time_sec']:<17.2f} | {r['infer_time_sec']:<16.2f} | "
            f"{str(r['final_train_loss']):<18} | {str(r['peak_train_mem_MB']):<20} | "
            f"{str(r['total_model_params']):<20} | {str(r['peak_infer_mem_MB']):<20}"
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
                "Final Train Loss": r["final_train_loss"],
                "Peak Train Mem (MB)": r["peak_train_mem_MB"],
                "Total Model Params": r["total_model_params"],
                "Peak Infer Mem (MB)": r["peak_infer_mem_MB"]
            })

    print("\n✅ Comparison results saved to ./exp/compare_results.csv")
