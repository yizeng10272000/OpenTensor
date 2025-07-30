import os
import time
import yaml
import glob

from codes.env import Environment
from codes.mcts import MCTS
from codes.net import Net
from codes.trainer import Trainer


def run_opentensor(use_projection=True, projection_dim=None):
    """
    Runs OpenTensor training + inference for either:
    - with tensor projection (with specified projection_dim)
    - without tensor projection
    """

    # === Load config ===
    with open("./config/S_4.yaml", "r") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)

    # === Override projection settings ===
    if use_projection:
        assert projection_dim is not None, "Must specify projection_dim for with_projection"
        kwargs["net"]["use_projection"] = True
        kwargs["net"]["projection_dim"] = projection_dim
        run_name = f"with_projection_{projection_dim}"
    else:
        kwargs["net"]["use_projection"] = False
        kwargs["net"].pop("projection_dim", None)
        run_name = "without_projection"

    # === Set experiment name ===
    kwargs["trainer"]["exp_name"] = run_name

    # === Check data exists ===
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
    trainer.learn(resume=None, example_path=data_path, self_example_path=None)
    t1 = time.time()
    train_time = t1 - t0
    print(f"[{run_name}] ✅ Training done in {train_time/60:.2f} mins")

    # === 2) Infer ===
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
    print("Starting OpenTensor Projection Comparison Run\n")

    # Ensure exp dir exists
    os.makedirs("./exp", exist_ok=True)

    # Prepare CSV file
    result_path = "./exp/compare_results.csv"
    with open(result_path, "w") as f:
        f.write("Run,ProjectionDim,TrainTime(min),InferTime(s),MCTSSteps\n")
        
    # Run without projection once
    result = run_opentensor(use_projection=False)
    with open(result_path, "a") as f:
        f.write(f"{result[0]},NA,{result[1]/60:.2f},{result[2]:.2f},{result[3]}\n")

    # Run with projection for various dimensions
    for proj_dim in range(16, 65, 8):  # 16, 24, 32, 40, 48， 56， 64
        result = run_opentensor(use_projection=True, projection_dim=proj_dim)
        with open(result_path, "a") as f:
            f.write(f"{result[0]},{proj_dim},{result[1]/60:.2f},{result[2]:.2f},{result[3]}\n")


    print("\n✅ All results saved to ./exp/compare_results.csv")
