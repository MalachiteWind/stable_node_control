from .utils import _load_wrapper

from scipy.integrate import solve_ivp
import numpy as np

def budworm_ode(t,x,r,k):
    return r*x*(1-x/k) - x**2 / (1+x**2)


def simulate_steady_state(k_vals, x0,show_progress:bool = True):

    wrapper = _load_wrapper(show_progress)
    t_span = [0,100]
    t_eval = [t_span[-1]]
    r=0.56
    x_vals = []
    x_curr = x0
    for k in wrapper(k_vals):
        sol = solve_ivp(budworm_ode, t_span=t_span, y0=[x_curr],args = (r,k,), t_eval=t_eval)
        x_curr = sol.y[0,-1]
        x_vals.append(x_curr)
    return np.array(x_vals)



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