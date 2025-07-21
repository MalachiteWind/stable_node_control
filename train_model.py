import time
import os
import argparse

import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np

from stabnode.node import FTerm, GTerm, set_global_seed, StabNODE, model_trainer
from data.hysteresis import simulate_steady_state
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'



def main(seed = 1234,sample_rate=10, lr=1e-2,n_epochs=100, hidden_dim = 2, patience=50,folder='results'):
    os.makedirs(folder, exist_ok=True)
    
    save_path = os.path.join(
        folder,
        f"seed_{seed}_sample_rate_{sample_rate}_lr_{lr}_n_epochs_{n_epochs}_hidden_dim_{hidden_dim}_patience_{patience}.pt"
    )

    # seed = 1234
    set_global_seed(seed = seed)
    
    lam_start = -1 
    lam_end = 1
    
    n_points = 250
    
    lam_increase = np.linspace(lam_start,lam_end, n_points)
    lam_decrease = np.linspace(lam_end, lam_start, n_points)
    
    x0=-1
    
    simulate_time = (0.,50.)
    
    x_increase = simulate_steady_state(lam_increase, x0, simulate_time,show_progress=False)
    x_decrease = simulate_steady_state(lam_decrease, x_increase[-1],simulate_time, show_progress=False)
    
    X = np.hstack((x_increase, x_decrease))
    lam = np.hstack((lam_increase, lam_decrease))
    repeat = 2
    
    for _ in range(repeat):
        X = np.hstack((X,X))
        lam = np.hstack((lam, lam))
    
    tau = torch.arange(0,len(lam),device=device)
    t = (tau+1)*simulate_time[-1]
    
    scaler = MinMaxScaler(feature_range=(-1,1))
    X_scaled = scaler.fit_transform(X.reshape(-1,1))
    X_scaled = torch.tensor(X_scaled,dtype=torch.float32, device=device)
    
    # sample_rate = 10
    tau_train = tau[::sample_rate]
    X_train = X_scaled[tau_train]
    lam_train = lam[tau_train]
    
    def lam_func(tau):
        idx = int(tau) % len(lam)
        return torch.tensor(lam[idx],dtype=torch.float32, device=device)
    
    dim_in = X_train.shape[-1]
    dim_out = X_train.shape[-1]
    # hidden_dim = 2
    
    f = FTerm(dim_in, dim_out, hidden_dim )
    g = GTerm(dim_in+1, dim_out, hidden_dim)
    
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
                  lam_func,
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
    parser.add_argument("--sample_rate", type=int, default=10, help="Sample rate")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--hidden_dim", type=int, default=4, help="Hidden layer dimension")
    parser.add_argument("--patience", type=int, default=50, help="patience on model trainer")
    parser.add_argument("--folder", type=str, default="results", help="Output folder")

    args = parser.parse_args()
    main(
        seed=args.seed, 
        sample_rate=args.sample_rate, 
        lr=args.lr, 
        n_epochs=args.n_epochs, 
        hidden_dim=args.hidden_dim, 
        patience = args.patience,
        folder=args.folder)