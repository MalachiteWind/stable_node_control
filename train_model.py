import time

import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np

from stabnode.node import FTerm, GTerm, set_global_seed, StabNODE
from data.hysteresis import simulate_steady_state
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 1234
set_global_seed(seed = seed)

lam_start = -1 
lam_end = 1

n_points = 250

lam_increase = np.linspace(lam_start,lam_end, n_points)
lam_decrease = np.linspace(lam_end, lam_start, n_points)

x0=-1

simulate_time = (0.,50.)

x_increase = simulate_steady_state(lam_increase, x0, simulate_time)
x_decrease = simulate_steady_state(lam_decrease, x_increase[-1],simulate_time)