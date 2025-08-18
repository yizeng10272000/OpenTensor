import yaml
import argparse
import numpy as np
import time
import random

from codes.env import Environment
from codes.mcts import MCTS
from codes.net import Net
from codes.trainer import Trainer, Player
from codes.utils.seed_utils import set_random_seed


def parse():
    parser = argparse.ArgumentParser(description="OpenTensor")
    parser.add_argument('--config', type=str, default="./config/S_4.yaml")
    parser.add_argument('--mode', type=str, default="train", help="modes: [generate_data, train, infer]")
    parser.add_argument('--resume', default=None, help="resume ckpt path for training")
    parser.add_argument('--run_dir', default=None, help="ckpt path for inference")
    parser.add_argument('--custom_tensor_path', type=str, default=None, help="Path to custom tensor .npy file for inference")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    conf_path = args.config
    mode = args.mode
    resume = args.resume

    with open(conf_path, 'r', encoding="utf-8") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # Set random seed
    seed = kwargs.get("seed", 42)
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Enable projection if specified
    if "projection_dim" in kwargs["net"]:
        kwargs["net"]["use_projection"] = True  
    else:
        kwargs["net"]["use_projection"] = False

    # Instantiate modules
    net = Net(**kwargs["net"])
    mcts = MCTS(**kwargs["mcts"], init_state=None)
    env = Environment(**kwargs["env"], init_state=None)
    trainer = Trainer(net=net, env=env, mcts=mcts, **kwargs["trainer"], all_kwargs=kwargs)

    S_size = kwargs["env"]["S_size"]
    T = kwargs["env"]["T"]
    domain = kwargs["env"].get("domain", "R")

    if mode == "generate_data":
        # If the domain is GF2, force the probability to [0.5, 0.5] to avoid ValueError
        prob = None
        if domain == "GF2":
            prob = [0.5, 0.5]

        trainer.generate_synthetic_examples(
            samples_n=100000,
            save_path="./data/100000_S%dT%d_scalar3_filtered.npy" % (S_size, T),
            domain=domain,
            prob=prob
        )

    elif mode == "train":
        trainer.learn(
            resume=resume,
            example_path="./data/100000_S%dT%d_scalar3_filtered.npy" % (S_size, T),
            self_example_path=None
        )

    elif mode == "infer":
        assert args.run_dir is not None, "Please specify --run_dir to the checkpoint you want to test!"

        # Load custom tensor if provided
        if args.custom_tensor_path is not None:
            custom_tensor = np.load(args.custom_tensor_path, allow_pickle=True)
            print("✅ Loaded custom tensor with shape:", custom_tensor.shape)
        else:
            custom_tensor = None

        t0 = time.time()
        step_ct = trainer.infer(resume=args.run_dir, tensor_override=custom_tensor)
        t1 = time.time()
        print(f"Infer done. Steps: {step_ct}, Time: {t1 - t0:.2f}s")

    else:
        raise ValueError(f"Unknown mode: {mode}")
