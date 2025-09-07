import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
from sympy import symbols, Eq, solve, simplify

# Local imports
from data.budworm import simulate_trials, budworm_steady_states
from stabnode.utils import set_global_seed
from stabnode.data import TrialsDataset
from stabnode.node import (
    GeluSigmoid,
    FeluSigmoid,
    StabNODE,
    model_trainer,
    FeluSigmoidMLP,
    GeluSigmoidMLP,
    GeluSigmoidMLPfeaturized,
    FeluSigmoidMLPfeaturized,
    _save_model_opt_cpu,
    _save_log_history
)


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main(args):
    # -----------------------------------------------------------------
    # CONFIGURATION
    # -----------------------------------------------------------------
    device = "cpu"
    set_global_seed(args.seed)

    # Simulation parameters
    k = 8.5

    # -----------------------------------------------------------------
    # DATA GENERATION
    # -----------------------------------------------------------------
    xs = []
    ts = []

    x0s = np.linspace(0.1,10,51)
    for x0 in x0s:
        budworm_trial =simulate_trials(
            [k],
            x0, 
            dt=0.1,
            r=0.56,
            eps=1e-3,
            buffer=1e-1,
            t_max=400,
            n_points=501, 
            show_progress=False
            )
        xs.append(budworm_trial.x_vals[0])
        ts.append(budworm_trial.t_vals[0])

    # -----------------------------------------------------------------
    # SCALING
    # -----------------------------------------------------------------
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate(xs).reshape(-1,1))

    xs_scaled = [scaler.transform(xi.reshape(-1,1)).reshape(-1) for xi in xs]

    # -----------------------------------------------------------------
    # TORCH DATASET PREP
    # -----------------------------------------------------------------
    xs_torch = [torch.tensor(xi,dtype=torch.float32,device=device) for xi in xs_scaled]
    ts_torch = [torch.tensor(ti,dtype=torch.float32,device=device) for ti in ts]
    ks_torch =[torch.tensor(k, dtype=torch.float32,device=device) for _ in xs]

    train_dataset = TrialsDataset(x_trials=xs_torch, t_trials=ts_torch,k_trials=ks_torch)
    train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=1,num_workers=0)

    # -----------------------------------------------------------------
    # MODEL DEFINITION
    # -----------------------------------------------------------------
    f = FeluSigmoidMLPfeaturized(
        dims = [4,10,10,1],
        activation = torch.nn.SiLU(), 
        lower_bound = -0.5, 
        upper_bound = -0.1,
        freq_sample_step=1,
        feat_lower_bound = 0,
        feat_upper_bound =1)


    g = GeluSigmoidMLPfeaturized(
        dims = [5,10,10,1], 
        activation = torch.nn.SiLU(), 
        lower_bound = -1.0183006535947712, 
        upper_bound = 0.6143380935111112, 
        freq_sample_step = 1,
        feat_lower_bound=0,
        feat_upper_bound=1
        )


    model = StabNODE(f,g).to(device)

    ## LOAD warmstart params if provided
    if args.model_config is not None:
        config = torch.load(args.model_config,map_location='cpu',weights_only=False)
        f.load_state_dict(config["f_state_dict"])
        g.load_state_dict(config["g_state_dict"])
        model.load_state_dict(config["stabnode_state_dict"])

    # -----------------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------------
    loss_criteria = nn.MSELoss()
    opt = torch.optim.Adam(list(f.parameters()) + list(g.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.1, patience=10
    )

    model, log_history = model_trainer(
        model=model,
        opt=opt,
        loss_criteria=loss_criteria,
        train_loader=train_loader,
        n_epochs=args.epochs,
        control=None,
        min_improvement=1e-6,
        patience=300,
        solve_method="tsit5",
        save_folder=args.save_folder,
        show_progress=True,
        scheduler=scheduler,
        print_every=10,
        _precision=10,
        train_dyn=True,
        decay_scheduler=None,
        decay_val=args.decay_val,
    )

    print("Training complete")
    if args.save_folder:
        print(f"Model/logs saved in: {args.save_folder}")
        base_path = Path(args.save_folder)
        base_path.mkdir(parents=True, exist_ok=True)

        model_opt_path_final = base_path / "model_opt_states_final.pt"

        _save_model_opt_cpu(
            model=model,
            opt=opt,
            epoch=args.epochs,
            loss=log_history["losses"][-1],
            save_path=model_opt_path_final,
            scheduler=scheduler
        )


    # return model, log_history



# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StabNODE on budworm dynamics")

    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs (default: 1000)")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning rate (default: 1e-2)")
    parser.add_argument("--save_folder", type=str, default=None,
                        help="Folder to save model and logs (default: None)")
    parser.add_argument("--decay_val", type=float, default=0.8,
                        help="Decay value for scheduler (default: 0.8)")
    
    parser.add_argument("--model_config", type=str,default=None,
                        help="Path to saved model config (.pt) for warm start.")

    parser.add_argument("--seed", type=int, default=1234,
                    help="Global randomness seed")

    args = parser.parse_args()
    main(args)
