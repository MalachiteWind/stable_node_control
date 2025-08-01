from tqdm.auto import tqdm 
import torch

def _load_loop_wrapper(show_progress:bool):
    if show_progress:
        return tqdm
    return lambda x: x 

def set_global_seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False