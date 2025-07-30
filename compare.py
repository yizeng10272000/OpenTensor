import os
import time
import yaml

from codes.env import Environment
from codes.mcts import MCTS
from codes.net import Net
from codes.trainer import Trainer


def run_opentensor(use_projection=True, projection_dim=None):
    """
    Runs OpenTensor training + inference for either:
    - with tensor projection (optionally with given projection_dim)
    - without tensor projection
    """
    # === Load config ===
    with open("./config/S_4.yaml", "r") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)

    # === Override projection ===
    if use_projection:
        run_name = f"with_projection_dim{projection_dim}"
        kwargs["net"]["use_projection"] = True
        kwargs["net"]["projection_dim"] = projection_dim
    else:
        run_name = "without_projection"
        kwargs["net"]["use_projection"] = False
        kwargs["net"].pop("projection_dim", None)

    # === Update experiment name ===
    kwargs["trainer"]["exp_name"] = run_name

    # === Check required data ===
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

    # === 2) Inference ===
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

    return (run_name, projection_dim, train_time, infer_time, steps)


if __name__ == "__main__":
    print("Starting OpenTensor Compare Run\n")

    results = []
    
    # === Without projection, only run once ===
    result_no_proj = run_opentensor(use_projection=False)
    results.append(result_no_proj)

    # === With projection, test multiple projection_dim values ===
#    for dim in range(16, 65, 8):  # 16, 24, 32, 40, 48, 56, 64
#        result = run_opentensor(use_projection=True, projection_dim=dim)
#        results.append(result)


    # === Print Summary ===
    print("\n=== Final Comparison Summary ===")
    print(f"{'Run':<30} {'ProjDim':<10} {'Train(min)':<12} {'Infer(s)':<10} {'MCTS Steps':<10}")
    print("-" * 70)
    for res in results:
        print(f"{res[0]:<30} {str(res[1]):<10} {res[2]/60:<12.2f} {res[3]:<10.2f} {res[4]:<10}")

    # === Save results to CSV ===
    os.makedirs("./exp", exist_ok=True)
    with open("./exp/compare_results.csv", "w") as f:
        f.write("Run,ProjectionDim,TrainTime(min),InferTime(s),MCTSSteps\n")
        for res in results:
            f.write(f"{res[0]},{res[1]},{res[2]/60:.2f},{res[3]:.2f},{res[4]}\n")

    print("\n✅ Comparison results saved to ./exp/compare_results.csv")
