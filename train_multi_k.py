import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from typing import List
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler

# Local imports
from data.budworm import simulate_trials, budworm_steady_states
from stabnode.utils import set_global_seed
from stabnode.node import (
    GeluSigmoid,
    FeluSigmoid,
    StabNODE,
    model_trainer,
    FeluSigmoidMLP,
    GeluSigmoidMLP,
    GeluSigmoidMLPfeaturized,
    FeluSigmoidMLPfeaturized,
)

# =====================================================================
# DATASET CLASS
# =====================================================================

class TrialsDataset(torch.utils.data.Dataset):
    def __init__(self, x_trials: List, t_trials: List, k_trials: List):
        self.x_trials = x_trials
        self.t_trials = t_trials
        self.k_trials = k_trials

    def __len__(self):
        return len(self.x_trials)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} is out of bounds of dataset size: {len(self)}."
            )

        Xi = self.x_trials[idx]
        ti = self.t_trials[idx]
        x0 = Xi[0]
        ki = self.k_trials[idx]

        return Xi, ti, x0, ki


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main(args):
    # -----------------------------------------------------------------
    # CONFIGURATION
    # -----------------------------------------------------------------
    device = "cpu"

    # Simulation parameters
    k_vals = [5.5, 6.3, 7.5, 8.5, 11.0]  # carrying capacities
    xs = np.linspace(0.1, 10, 51)

    # -----------------------------------------------------------------
    # DATA GENERATION
    # -----------------------------------------------------------------
    trials = {}
    for idx, ki in enumerate(k_vals):
        ki_traj, ki_times = [], []

        for x0 in xs:
            trial = simulate_trials(
                [ki],
                x0,
                dt=0.2,
                r=0.56,
                eps=1e-3,
                buffer=1e-1,
                t_max=400,
                n_points=501,
                show_progress=False,
            )
            if len(trial.t_vals[0]) == 1:
                continue
            ki_traj.append(trial.x_vals[0])
            ki_times.append(trial.t_vals[0])

        trials[str(idx)] = {
            "traj": ki_traj,
            "times": ki_times,
            "k": ki,
            "trial": idx,
        }

    trial_df = pd.DataFrame(trials)

    # -----------------------------------------------------------------
    # SCALING
    # -----------------------------------------------------------------
    full_traj = [np.concatenate(trial_df[col].traj) for col in trial_df.columns]
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate(full_traj).reshape(-1, 1))

    scaled_rows = {}
    for col in trial_df.columns:
        traj = trial_df[col].traj
        scaled_rows[col] = [
            scaler.transform(xi.reshape(-1, 1)).reshape(-1) for xi in traj
        ]
    trial_df.loc["scaled_traj"] = pd.Series(scaled_rows)

    # -----------------------------------------------------------------
    # TORCH DATASET PREP
    # -----------------------------------------------------------------
    all_xs, all_ts, all_ks = [], [], []

    for trial_i in trial_df.columns:
        xs_scaled = trial_df[trial_i].scaled_traj
        ts = trial_df[trial_i].times
        k = trial_df[trial_i].k

        xs_torch = [
            torch.tensor(xi, dtype=torch.float32, device=device) for xi in xs_scaled
        ]
        ts_torch = [
            torch.tensor(ti, dtype=torch.float32, device=device) for ti in ts
        ]
        ks = [torch.tensor(k, dtype=torch.float32, device=device) for _ in ts]

        all_xs.extend(xs_torch)
        all_ts.extend(ts_torch)
        all_ks.extend(ks)

    train_dataset = TrialsDataset(all_xs, all_ts, all_ks)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=1, num_workers=0
    )

    # -----------------------------------------------------------------
    # MODEL DEFINITION
    # -----------------------------------------------------------------
    f = FeluSigmoidMLP(
        dims=[1, 10, 10, 1],
        activation=nn.SiLU(),
        lower_bound=-0.5,
        upper_bound=-0.1,
    )

    g = GeluSigmoidMLPfeaturized(
        dims=[6, 10, 10, 1],
        activation=nn.SiLU(),
        lower_bound=-5,
        upper_bound=1.1,
        freq_sample_step=1,
        feat_lower_bound=0,
        feat_upper_bound=1,
    )

    model = StabNODE(f, g).to(device)

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

    print("Training complete ✅")
    if args.save_folder:
        print(f"Model/logs saved in: {args.save_folder}")
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

    args = parser.parse_args()
    main(args)
