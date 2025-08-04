from .utils import _load_wrapper
from typing import Tuple, Optional

from scipy.integrate import solve_ivp
import numpy as np
import torch


def hysteresis_ode(t,x,lam):
    return lam+x-x**3
    
def simulate_steady_state(
    lam_values: np.ndarray, 
    x0: float, 
    t_span:Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    show_progress: bool = True,
    device:str = 'cpu'
)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Arguments:
    ----------
    lam_values: control params 
    x0: initial state observation
    t_span: start and stop time of ode solution 
    t_eval: Points at which to save results. Default is end of t_span.
    show_progress: Show progress bar (tqdm)
    device: where to excute computation. 'cuda', 'cpu', etc.
    
    Returns:
    --------
    x_vals: Output tensor of dataset (n_samples, n_dim)
    lams_vals: control tensor at associated time (n_samples,)
    t_vals: assoicated time tensor. (n_samples, )
    """
    wrapper = _load_wrapper(show_progress)

    x_curr = np.array([x0])
    
    if t_eval is None:
        t_eval = np.array([t_span[-1]])
    
    for idx, lam in enumerate(wrapper(lam_values)):
        sol = solve_ivp(
            hysteresis_ode, 
            t_span=t_span, 
            y0=x_curr, 
            args=(lam,), 
            t_eval=t_eval
        )

        x_curr = sol.y[:, -1]  # final value

        lam_i = np.array([lam]*len(t_eval))

        
        if idx == 0:
            x_vals = sol.y
            lam_vals = lam_i
            t_vals = t_eval
        
        else:
            shift=1
            if sol.y.shape[-1]==1:
                shift = 0

            x_vals = np.hstack((x_vals, sol.y[:,shift:]))
            lam_vals = np.hstack((lam_vals,lam_i[shift:]))
            t_vals = np.hstack((t_vals,t_eval[shift:] + t_eval[-1]*idx))

    x_vals = torch.tensor(x_vals, dtype=torch.float64,device=device)
    lam_vals = torch.tensor(lam_vals, dtype=torch.float64,device=device)    
    t_vals = torch.tensor(t_vals, dtype=torch.float64,device=device)
    return x_vals, lam_vals, t_vals