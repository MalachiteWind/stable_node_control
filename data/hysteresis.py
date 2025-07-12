from utils import _load_wrapper

def hysteresis_ode(t,x,lam):
    return lam+x-x**3
    
def simulate_steady_state(lam_values, x0, show_progress: bool = True):
    x_vals = []
    x_curr = x0

    wrapper = _load_wrapper(show_progress)
    
    for lam in wrapper(lam_values):
        sol = solve_ivp(hysteresis_ode, [0, 50], [x_curr], args=(lam,), t_eval=[50])
        x_curr = sol.y[0, -1]  # final value
        x_vals.append(x_curr)
        
    return np.array(x_vals)