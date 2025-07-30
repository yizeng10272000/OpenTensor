# codes/seed_utils.py

import os
import random
import numpy as np
import torch

def set_random_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for reproducibility across random, numpy, torch, and cuda.

    Args:
        seed (int): The random seed to use.
        deterministic (bool): Whether to force deterministic behavior in CUDA.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
