from .utils import _load_wrapper
from typing import Tuple

from scipy.integrate import solve_ivp
import numpy as np


def hysteresis_ode(t,x,lam):
    return lam+x-x**3
    
def simulate_steady_state(lam_values, x0, simulate_time:Tuple[float, float],show_progress: bool = True):
    """
    lam_values: control params 
    x0: initial state observation
    simulate_time: start and stop time of ode solution 
    
    """
    
    x_vals = []
    x_curr = x0

    wrapper = _load_wrapper(show_progress)
    
    for lam in wrapper(lam_values):
        sol = solve_ivp(hysteresis_ode, simulate_time, [x_curr], args=(lam,), t_eval=[simulate_time[-1]])
        x_curr = sol.y[0, -1]  # final value
        x_vals.append(x_curr)
        
    return np.array(x_vals)