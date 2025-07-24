import os
import time
import yaml
import glob

from codes.env import Environment
from codes.mcts import MCTS
from codes.net import Net
from codes.trainer import Trainer


def run_opentensor(use_projection=True):
    """
    Runs OpenTensor training + inference for either:
    - with tensor projection
    - without tensor projection
    """

    # === Load config ===
    with open("./config/S_4.yaml", "r") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)

    # === Override projection ===
    kwargs["net"]["use_projection"] = use_projection
    if use_projection:
        kwargs["net"]["projection_dim"] = kwargs["net"].get("projection_dim", 16)
        run_name = "with_projection"
    else:
        kwargs["net"].pop("projection_dim", None)
        run_name = "without_projection"

    # === Update experiment name to separate runs ===
    kwargs["trainer"]["exp_name"] = run_name

    # === Check required data exists ===
    data_path = "./data/100000_S%dT%d_scalar3_filtered.npy" % (
        kwargs["env"]["S_size"], kwargs["env"]["T"]
    )
    assert os.path.exists(data_path), f"Required data not found: {data_path}"

    # === Build components ===
    net = Net(**kwargs["net"])
    mcts = MCTS(**kwargs["mcts"], init_state=None)
    env = Environment(**kwargs["env"], init_state=None)
    trainer = Trainer(**kwargs["trainer"], net=net, env=env, mcts=mcts, all_kwargs=kwargs)

    # === 1) Train ===
    print(f"\n=== Training [{run_name}] ===")
    t0 = time.time()
    trainer.learn(
        resume=None,
        example_path=data_path,
        self_example_path=None
    )
    t1 = time.time()
    train_time = t1 - t0

    print(f"[{run_name}] ✅ Training done in {train_time/60:.2f} mins")

    # === 2) Infer ===
    # === Automatically find the latest random ID subfolder ===
    exp_base = f"./exp/{run_name}"
    all_subdirs = [os.path.join(exp_base, d) for d in os.listdir(exp_base)
                   if os.path.isdir(os.path.join(exp_base, d))]

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

    print(f"[{run_name}] ✅ Inference done in {infer_time:.2f}s | MCTS steps: {steps}")

    return (run_name, train_time, infer_time, steps)


if __name__ == "__main__":
    print(" Starting OpenTensor Compare Run\n")

    result1 = run_opentensor(use_projection=True)
    result2 = run_opentensor(use_projection=False)

    print("\n=== Final Comparison Summary ===")
    print(f"{'Run':<20} {'Train(min)':<12} {'Infer(s)':<10} {'MCTS Steps':<10}")
    print("-" * 50)
    for res in [result1, result2]:
        print(f"{res[0]:<20} {res[1]/60:<12.2f} {res[2]:<10.2f} {res[3]:<10}")

    # Ensure exp dir exists
    os.makedirs("./exp", exist_ok=True)

    # Save CSV
    with open("./exp/compare_results.csv", "w") as f:
        f.write("Run,TrainTime(min),InferTime(s),MCTSSteps\n")
        for res in [result1, result2]:
            f.write(f"{res[0]},{res[1]/60:.2f},{res[2]:.2f},{res[3]}\n")

    print("\n✅ Comparison results saved to ./exp/compare_results.csv")
