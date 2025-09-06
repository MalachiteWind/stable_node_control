import argparse
import time
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torchode import solve_ivp
from pathlib import Path
from typing import List, Optional, Callable, Tuple
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler

# Local imports
from data.budworm import simulate_trials
from stabnode.utils import set_global_seed, _load_loop_wrapper
from stabnode.node import (
    _create_save_paths,
    _save_log_history,
    MLP
)
from stabnode.schedulers import ExpLossTimeDecayScheduler

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
# New Model Trainer
# =====================================================================
class ODEFunc(nn.Module):
    def __init__(
        self,
        dims,          
        activation=nn.SiLU(),
        lower_bound=-1.0,
        upper_bound=1.0,
        dtype=torch.float
    ):
        super().__init__()
        self.dims = dims
        self.activation = activation
        self.dtype = dtype

        self.network = MLP(self.dims, activation=self.activation, dtype=self.dtype)

        self.args = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

    def forward(self,x,k):
        """
        state = (x, k)
        x : [batch, dim_x]
        k : [batch, dim_k]
        """
        # x, k 
        a = self.args["lower_bound"]
        b = self.args["upper_bound"]

        xk = torch.cat([x, k], dim=-1)                 
        dxdt = a + (b - a) * torch.sigmoid(self.network(xk))  

        return dxdt


def _save_model_opt_cpu(model: ODEFunc, opt, epoch, loss, save_path: str, scheduler=None):
    device = next(model.parameters()).device.type
    
    # Move to CPU if necessary
    if device == "cpu":
        model_state = model.state_dict()
    else:
        model_cpu = copy.deepcopy(model).to("cpu")
        model_state = model_cpu.state_dict()

    # Save constructor args
    model_args = {
        "dims": model.dims,
        "activation": type(model.activation),  # store class, not instance
        "lower_bound": model.args["lower_bound"],
        "upper_bound": model.args["upper_bound"],
        "dtype": model.dtype,
    }

    checkpoint = {
        "model_state_dict": model_state,
        "model_args": model_args,
        "opt_state_dict": opt.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)


def model_trainer(
        model: ODEFunc,
        opt: torch.optim.Optimizer,
        loss_criteria: Callable,
        train_loader: torch.utils.data.DataLoader,
        n_epochs: int,
        control: Callable[[torch.Tensor], torch.Tensor],
        min_improvement:float,
        patience: int,
        solve_method: str='tsit5', 
        save_folder: str|Path=None,
        show_progress:bool=True,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None,
        print_every: int=5,
        _precision: int = 4,
        effective_batch_size: int = 10,
        train_dyn = True,
        decay_scheduler: Optional[ExpLossTimeDecayScheduler] = None,
        decay_val:int = 0.0
)-> Tuple[ODEFunc,dict]:
    """
    if decay_scheduler is given, this takes priority over decay_val.
    """
    
    loop_wrapper = _load_loop_wrapper(show_progress)
    model_opt_save_path, log_save_path = _create_save_paths(save_folder)

    best_loss = torch.inf
    patience_count = 0
    best_model_epoch = -1
    stopping_criteria = 'max-epochs'

    losses = []
    times = []
    status = []
    patience_hist = []
    lr_hist = []
    alpha_hist = []
    model.train()
    max_iters = len(train_loader)
     #this is training iteration counter to keep track of effective batch size.
    for epoch in loop_wrapper(range(n_epochs)):
        t1 = time.time()
        epoch_loss = 0.0
        num_batches = 0
        epochs_status = []
        iter_counter = 0
        for Xi, Ti, x0i, ki in train_loader:
            Xi = Xi.squeeze() # [batch, time, dim]
            Ti = Ti.squeeze()
            x0i = x0i.view(-1,1)

            # if not x0i.requires_grad:
            #     x0i = x0i.clone().detach().requires_grad_()

            control = lambda t: ki

            opt.zero_grad()

            def rhs(t,x):
                return model(x,ki.view(-1,1))

            if train_dyn == True:
                sol = solve_ivp(
                    f=rhs,
                    y0=x0i,
                    t_eval=Ti,
                    method=solve_method
                )

                epochs_status.append(sol.status)

                if decay_scheduler is not None:
                    decay_val = decay_scheduler.get_alpha() 

                Xi_pred = sol.ys.squeeze()
                loss = loss_criteria(
                    Xi_pred*torch.exp(-decay_val*Ti), 
                    Xi*torch.exp(-decay_val*Ti)
                )

            Xi = Xi.unsqueeze(-1)
            cntrl = control(Ti)
            cntrl = torch.reshape(cntrl,(1,1))
            cntrl = cntrl.repeat(Xi.shape[0],1)
            g_id_loss = 100*loss_criteria(model(Xi,cntrl),Xi)

            if train_dyn == True:
                #loss = loss +  g_id_loss
                loss = loss
            else:
                loss = g_id_loss

            loss.backward()

            iter_counter += 1
            if effective_batch_size>= 1:
                if (iter_counter+1)%effective_batch_size==0 or iter_counter>= max_iters:
                    opt.step()
                    opt.zero_grad()
                    num_batches += 1
            else:
                opt.step()
                opt.zero_grad()
                num_batches += 1

            epoch_loss+= loss.item()
        epoch_loss = epoch_loss / num_batches

        if decay_scheduler is not None:
            decay_scheduler.step(epoch_loss)

            if decay_scheduler.get_alpha() == 0.0 and scheduler is not None:
                scheduler.step(epoch_loss)
                
        elif scheduler is not None:
            scheduler.step(epoch_loss)
     
        cur_lr = opt.param_groups[0]['lr']
        cur_alpha = decay_scheduler.get_alpha() if decay_scheduler is not None else decay_val

        epoch_time = time.time() - t1

        losses.append(epoch_loss)
        times.append(epoch_time)
        status.append(epochs_status)
        lr_hist.append(cur_lr)

        if show_progress:
            if epoch <= 5 or epoch % print_every == 0 or epoch == n_epochs-1:
                print(f"Epoch {epoch}: Loss: {epoch_loss:.{_precision}f}. time = {epoch_time:.{_precision}f}s. lr = {cur_lr:.{_precision}f}. alpha = {cur_alpha:.{_precision}f}")    
        
        # model checks
        if best_loss - epoch_loss >= min_improvement:
            best_loss = epoch_loss
            patience_count = 0
            best_model_epoch = epoch

            if save_folder is not None:
                _save_model_opt_cpu(
                    model,
                    opt,
                    best_model_epoch,
                    best_loss,
                    model_opt_save_path,
                    scheduler
                )

        else:
            patience_count += 1
        
        patience_hist.append(patience_count)

        if patience_count > patience:
            stopping_criteria = 'early-stoppage'
            if show_progress is not None:
                print(f"Patience exceeded: {patience}. Early stoppage executed.")
            break
        
        if save_folder is not None:
            _ = _save_log_history(
                losses,
                times,
                stopping_criteria=f"checkpoint-{epoch}",
                best_model_epoch=best_model_epoch,
                method_status=status,
                patience_hist=patience_hist,
                lr_hist=lr_hist,
                save_path = log_save_path
            )
        
    log_history = _save_log_history(
        losses,
        times,
        stopping_criteria,
        best_model_epoch,
        status,
        patience_hist,
        lr_hist,
        log_save_path,
    )

    return model, log_history

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

    k_vals = [5.5, 8.5,11,7.5,6.3]
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

    model = ODEFunc(
        dims=[2,10,10,1],
        activation=nn.SiLU(),
        lower_bound=0,
        upper_bound=1
    ).to(device)

    # -----------------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------------
    loss_criteria = nn.MSELoss()
    opt = torch.optim.Adam(list(model.parameters()), lr=args.lr)
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

    parser.add_argument("--seed", type=int, default=1234,
                    help="Global randomness seed")

    args = parser.parse_args()
    main(args)
