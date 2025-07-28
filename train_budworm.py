import time
import os
import argparse

import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np

from sympy import symbols, Eq, solve, simplify
from stabnode.node import Felu, Gelu, set_global_seed, StabNODE, model_trainer
from data.budworm import simulate_steady_state
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'

def get_roots():
    """
    roots of descriminant of associated ode cubic in k.
    """
    r, k = symbols('r k', positive=True)
    a = r/k
    b = -r
    c = (k+r)/k
    d = -r
    p = (3*a*c-b**2) / (3*a**2)
    q = (2*b**3 - 9*a*b*c+27*a**2*d) / (27*a**3)
    
    D = - (4*p**3 + 27*q**2)
    D = simplify(D)
    
    
    D_fixed = D.subs(r, 0.56)
    r1, r2 = solve(Eq(D_fixed, 0), k)
    return r1, r2


def main(seed = 1234,noise=0.0,sample_rate=10, lr=1e-2,n_epochs=100, f_dim = 2,g_dim=2, patience=50,folder='results'):
    os.makedirs(folder, exist_ok=True)
    
    save_path = os.path.join(
        folder,
        f"seed_{seed}_noise_{noise}_sample_rate_{sample_rate}_lr_{lr}_n_epochs_{n_epochs}_f_dim_{f_dim}_g_dim_{g_dim}_patience_{patience}.pt"
    )

    # seed = 1234
    set_global_seed(seed = seed)

    r1, r2 = get_roots()
    k_vals = np.linspace(float(r1)-1, float(r2)+1,250)


    x0 = 1

    k_start = k_vals[0]
    k_end = k_vals[-1]
    n_points = 250
    
    k_increase = np.linspace(k_start,k_end, n_points)
    k_decrease = np.linspace(k_increase[-1],k_start,n_points)

    x_increase = simulate_steady_state(k_increase,x0=x0, show_progress=False)
    x_decrease = simulate_steady_state(k_decrease, x0=x_increase[-1], show_progress=False)

    X = np.hstack((x_increase,x_decrease))
    K = np.hstack((k_increase, k_decrease))

    repeat = 2
    for _ in range(repeat):
        X = np.hstack((X,X))
        K = np.hstack((K,K))
    
    tau = torch.arange(0, len(K), device = device)

        
    scaler = MinMaxScaler(feature_range=(-1,1))
    X_scaled = scaler.fit_transform(X.reshape(-1,1))
    X_scaled = torch.tensor(X_scaled,dtype=torch.float32, device=device)

    tau_train = tau[::sample_rate]
    X_train = X_scaled[tau_train]
    K_train = K[tau_train]

    noise_arr = torch.randn_like(X_train) * noise
    X_train = X_train + noise_arr

    def K_func(tau):
        idx = int(tau) % len(K)
        return torch.tensor(K[idx],dtype=torch.float32, device=device) 


    dim_in = X_train.shape[-1]
    dim_out = X_train.shape[-1]

    f = Felu(dim_in, dim_out, f_dim)
    g = Gelu(dim_in+1, dim_out, g_dim)

    stab_node = StabNODE(f,g)
    stab_node.to(device)
    
    loss_criteria = nn.MSELoss()
    opt = torch.optim.Adam(list(f.parameters())+list(g.parameters()), lr = lr)
    tau_span = tau_train.clone().detach().to(dtype=torch.float32, device=device)
    x0 = X_train[0].reshape(-1, 1).clone().detach().to(dtype=torch.float32,device=device)
    x0.requires_grad_(True)
    # n_epochs = 100
    
    stab_node, log_history = model_trainer(
                  stab_node,
                  opt,
                  loss_criteria,
                  x0,
                  tau_span,
                  X_train,
                  K_func,
                  n_epochs,
                  min_improvement=1e-4,
                  patience=patience,
                  print_every=10,
                  solve_method='dopri5',
                  show_progress=False,
                  save_path=save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--noise", type=float, default=0.0, help="Training noise")
    parser.add_argument("--sample_rate", type=int, default=10, help="Sample rate")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--f_dim", type=int, default=2, help="f hidden layer dimension")
    parser.add_argument("--g_dim", type=int, default=2, help="g hidden layer dimension")
    parser.add_argument("--patience", type=int, default=50, help="patience on model trainer")
    parser.add_argument("--folder", type=str, default="results", help="Output folder")

    args = parser.parse_args()
    main(
        seed=args.seed, 
        noise=args.noise,
        sample_rate=args.sample_rate, 
        lr=args.lr, 
        n_epochs=args.n_epochs, 
        f_dim=args.f_dim,
        g_dim = args.g_dim,
        patience = args.patience,
        folder=args.folder)