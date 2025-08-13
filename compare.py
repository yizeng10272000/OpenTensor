import os
import time
import yaml
import csv
import re

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

    final_train_loss = getattr(trainer, "final_train_loss", "NA")
    peak_train_mem = getattr(trainer, "peak_train_memory_MB", "NA")

    total_params = sum(p.numel() for p in net.parameters())

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
    final_rank = getattr(trainer, "final_rank", "NA")
    per_step_log = getattr(trainer, "per_step_log", [])

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
        "final_rank": final_rank,
        "latest_subdir": latest_subdir,
        "per_step_log": per_step_log,
    }


def parse_infer_txt_all_blocks(txt_path):
    pattern = re.compile(
        r"Depth:\s*\n\s*(\d+).*?"
        r"scores:\s*\n\s*\[([^\]]+)\].*?"
        r"Q:\s*\n\s*\[([^\]]+)\].*?"
        r"N:\s*\n\s*\[([^\]]+)\]",
        re.DOTALL
    )

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    matches = pattern.findall(content)
    results = []
    for depth, scores, q, n in matches:
        score_list = [s.strip() for s in scores.strip().split()]
        q_list = [qv.strip() for qv in q.strip().split()]
        n_list = [nv.strip() for nv in n.strip().split()]
        results.append({
            "Depth": int(depth),
            "Scores": score_list,
            "Q": q_list,
            "Action Count": n_list,
        })
    return results


def read_infer_txt_records(latest_subdir):
    infer_dir = os.path.join(latest_subdir, "infer")
    if not os.path.exists(infer_dir):
        return []

    txt_files = [f for f in os.listdir(infer_dir) if f.endswith(".txt")]
    if not txt_files:
        return []

    txt_path = os.path.join(infer_dir, txt_files[0])
    try:
        return parse_infer_txt_all_blocks(txt_path)
    except Exception as e:
        print(f"⚠️ Failed to parse {txt_path}: {e}")
        return []


if __name__ == "__main__":
    print("Starting OpenTensor Compare Run\n")

    results = []

    # Run with projection
    # for dim in range(16, 61, 4):
    #    res = run_opentensor(use_projection=True, projection_dim=dim)
    #    results.append(res)
        
    # Run without projection
    result_no_proj = run_opentensor(use_projection=False)
    results.append(result_no_proj)

    max_len = 5
    header = (
        ["Run Name", "Use Projection", "Projection Dim",
         "Train Time (sec)", "Infer Time (sec)",
         "Final Train Loss", "Peak Train Mem (MB)",
         "Total Model Params", "Peak Infer Mem (MB)",
         "Final Rank", "Depth",
         "Step Infer Time (sec)", "Step Peak Mem (MB)", "Step Pi Max"]
        + [f"Score_{i}" for i in range(max_len)]
        + [f"Q_{i}" for i in range(max_len)]
        + [f"Action Count_{i}" for i in range(max_len)]
    )

    os.makedirs("./exp", exist_ok=True)
    with open("./exp/compare_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for r in results:
            infer_records = read_infer_txt_records(r["latest_subdir"])
            step_logs = r.get("per_step_log", [])
            step_log_dict = {entry["step"]: entry for entry in step_logs}

            if infer_records:
                for rec in infer_records:
                    depth = int(rec["Depth"])
                    step_info = step_log_dict.get(depth, {})
                    row = {
                        "Run Name": r["run_name"],
                        "Use Projection": r["use_projection"],
                        "Projection Dim": r["projection_dim"],
                        "Train Time (sec)": f"{r['train_time_sec']:.2f}",
                        "Infer Time (sec)": f"{r['infer_time_sec']:.2f}",
                        "Final Train Loss": r["final_train_loss"],
                        "Peak Train Mem (MB)": r["peak_train_mem_MB"],
                        "Total Model Params": r["total_model_params"],
                        "Peak Infer Mem (MB)": r["peak_infer_mem_MB"],
                        "Final Rank": r["final_rank"],
                        "Depth": depth,
                        "Step Infer Time (sec)": f"{step_info.get('step_infer_time_sec', ''):.4f}" if "step_infer_time_sec" in step_info else "",
                        "Step Peak Mem (MB)": f"{step_info.get('step_peak_mem_MB', ''):.2f}" if "step_peak_mem_MB" in step_info else "",
                        "Step Pi Max": f"{step_info.get('step_pi_max', ''):.4f}" if "step_pi_max" in step_info else "",
                    }

                    for i in range(max_len):
                        row[f"Score_{i}"] = rec["Scores"][i] if i < len(rec["Scores"]) else ""
                        row[f"Q_{i}"] = rec["Q"][i] if i < len(rec["Q"]) else ""
                        row[f"Action Count_{i}"] = rec["Action Count"][i] if i < len(rec["Action Count"]) else ""

                    writer.writerow(row)
            else:
                writer.writerow({
                    "Run Name": r["run_name"],
                    "Use Projection": r["use_projection"],
                    "Projection Dim": r["projection_dim"],
                    "Train Time (sec)": f"{r['train_time_sec']:.2f}",
                    "Infer Time (sec)": f"{r['infer_time_sec']:.2f}",
                    "Final Train Loss": r["final_train_loss"],
                    "Peak Train Mem (MB)": r["peak_train_mem_MB"],
                    "Total Model Params": r["total_model_params"],
                    "Peak Infer Mem (MB)": r["peak_infer_mem_MB"],
                    "Final Rank": r["final_rank"],
                    "Depth": ""
                })

    print("\n✅ Comparison results saved to ./exp/compare_results.csv with per-step inference details.")
