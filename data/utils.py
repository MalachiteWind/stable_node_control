from tqdm.auto import tqdm 

def _load_wrapper(show_progres:bool):
    if show_progress:
        return tqdm
    return lambda x: x 