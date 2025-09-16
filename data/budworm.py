from .utils import _load_wrapper
from typing import Tuple, Optional, List

from scipy.integrate import solve_ivp
from dataclasses import dataclass
import numpy as np
import torch


def budworm_ode(t,x,r,k):
    return r*x*(1-x/k) - x**2 / (1+x**2)

def f_true(x):
    return -1*x / (1+x**2)

def g_true(x,k,r):
    """
    g(x,k;r) = (r/k)*(1+x^2)(k-x)
    """
    return -(r/k)*x**3 + r*x**2 -(r/k)*x +r 

def dg_true(x,k,r):
    return -3*(r/k)*x**2 + 2*r*x - (r/k)

def budworm_steady_states(k,r):
    """
    for a given k and r return real x such that 
    g(x,k:r) = x

    find roots of polynomial 
    p(x) = g(x,k;r)-x = -r/k x^3 + rx^2 -(r/k+1)x+r
    """
    a = -r/k
    b = r
    c=-(r/k + 1)
    d = r
    roots = np.roots([a,b,c,d])

    return sorted([r.real for r in roots if np.isreal(r)])

def converging_steady_state(steady_states,x0):
    """
    Based on initial condtion x0, return steady state
    that budworm system will converge to. 
    """
    if len(steady_states)==3:
        if x0 < steady_states[1]:
            return steady_states[0]
        return steady_states[-1]
    return steady_states[0]


@dataclass
class BudwormTrials:
    x_vals: List
    t_vals: List
    k_vals: List
    x_stars: List
    t_stars: List
    indices: List
    dt: float

    def __str__(self):
        attrs = {
            "x_vals": self.x_vals,
            "t_vals": self.t_vals,
            "k_vals": self.k_vals,
            "x_stars": self.x_stars,
            "t_stars": self.t_stars,
            "indices": self.indices
        }
        lines = ["BudwormTrials contents:"]
        for key, val in attrs.items():
            lines.append(f"  {key}: (len={len(val)})")
        lines.append(f"  dt: {self.dt}")
        return "\n".join(lines)

    __repr__ = __str__  


def simulate_trials(ks, x0, dt, r,eps,buffer, t_max, n_points,show_progress:bool=True):
    x_curr = x0

    x_vals = []
    t_vals = []
    x_stars = []
    t_stars = []
    indices = []

    loop_wrapper = _load_wrapper(show_progress)
    for k in loop_wrapper(ks):
        # determine steady state
        xs = budworm_steady_states(k,r)
        x_star = converging_steady_state(xs,x_curr)

        # find time to steady state
        sol = solve_ivp(
            budworm_ode,
            t_span=[0,t_max],
            y0=np.array([x_curr]),
            t_eval=np.linspace(0,t_max,n_points),
            args=(r,k)
        )

        idx_star = np.where(np.abs(sol.y[0,:] - x_star)<eps)[0][0]
        
        t_star = sol.t[idx_star]


        t_end = t_star + buffer
        t_span = [0,t_end]
        t_eval = np.arange(0,t_end, dt)

        
        # Run simuilation until steady state
        sol = solve_ivp(
            fun=budworm_ode,
            t_span=t_span,
            y0=np.array([x_curr]),
            t_eval=t_eval,
            args=(r,k)
        )
        x_curr = sol.y[0,-1]
        
        indices.append(idx_star)
        x_vals.append(sol.y[0,:])
        t_vals.append(sol.t)
        x_stars.append(x_star)
        t_stars.append(t_star)
    return BudwormTrials(
        x_vals,t_vals,ks,x_stars, t_stars, indices,dt
    )

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