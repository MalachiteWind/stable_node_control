from tqdm.auto import tqdm 
import torch
import numpy as np
import random

def _load_loop_wrapper(show_progress:bool):
    if show_progress:
        return tqdm
    return lambda x: x 

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # check to see if any torch operatiosn introduce randomness
    torch.use_deterministic_algorithms(True, warn_only=True)