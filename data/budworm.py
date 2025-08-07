from .utils import _load_wrapper
from typing import Tuple, Optional

from scipy.integrate import solve_ivp
import numpy as np
import torch

def budworm_ode(t,x,r,k):
    return r*x*(1-x/k) - x**2 / (1+x**2)


def simulate_steady_state(
    k_values: np.ndarray, 
    x0: float, 
    t_span:Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    show_progress: bool = True,
    device:str = 'cpu',
    dtype:Optional[torch.dtype]=None
)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Arguments:
    ----------
    k_values: control params 
    x0: initial state observation
    t_span: start and stop time of ode solution 
    t_eval: Points at which to save results. Default is end of t_span.
    show_progress: Show progress bar (tqdm)
    device: where to excute computation. 'cuda', 'cpu', etc.
    
    Returns:
    --------
    x_vals: Output tensor of dataset (n_samples, n_dim)
    k_vals: control tensor at associated time (n_samples,)
    t_vals: assoicated time tensor. (n_samples, )
    """
    wrapper = _load_wrapper(show_progress)

    x_curr = np.array([x0])
    r=0.56
    if t_eval is None:
        t_eval = np.array([t_span[-1]])
    
    for idx, k in enumerate(wrapper(k_values)):
        sol = solve_ivp(
            budworm_ode, 
            t_span=t_span, 
            y0=x_curr, 
            args=(r,k,), 
            t_eval=t_eval
        )

        x_curr = sol.y[:, -1]  # final value

        k_i = np.array([k]*len(t_eval))

        
        if idx == 0:
            x_vals = sol.y
            k_vals = k_i
            t_vals = t_eval
        
        else:
            shift=1
            if sol.y.shape[-1]==1:
                shift = 0

            x_vals = np.hstack((x_vals, sol.y[:,shift:]))
            k_vals = np.hstack((k_vals,k_i[shift:]))
            t_vals = np.hstack((t_vals,t_eval[shift:] + t_eval[-1]*idx))
    if dtype is None:
        dtype = torch.float32

    x_vals = torch.tensor(x_vals, dtype=dtype,device=device)
    k_vals = torch.tensor(k_vals, dtype=dtype,device=device)    
    t_vals = torch.tensor(t_vals, dtype=dtype,device=device)
    return x_vals, k_vals, t_vals


def descriminant(k,r):
    """
    ax^3 + bx^2 + cx + d = 0
    https://en.wikipedia.org/wiki/Cubic_equation
    """
    a = r/k
    b = -r
    c = (k+r)/k
    d = -r
    p = (3*a*c-b**2) / (3*a**2)
    q = (2*b**3 - 9*a*b*c+27*a**2*d) / (27*a**3)

    return - (4*p**3 + 27*q**2)