import argparse
import time
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torchode
from scipy.integrate import solve_ivp as sp_solve_ivp

from stabnode.utils import set_global_seed, _load_loop_wrapper
from stabnode.data import TrialsDataset
from stabnode.node import (
    StabNODE, FeluSigmoidMLP, GeluSigmoidMLP,
    _create_save_paths, _save_model_opt_cpu, _save_log_history
)
from stabnode.schedulers import ExpLossTimeDecayScheduler



## Define system 
gamma = 50
def sigmoid(x,gamma=gamma):
    return 1 / (1+np.exp(-gamma*x))

eps = 0.02
q1, q2 = (0.08, 0.04)
b1 = 1-eps
b2 = 1-eps

def c1_in(x):
    return q1*(1-sigmoid(x-b1))

def c2_in(y):
    return q1*(1-sigmoid(y-b2))

def c1_out(y):
    return q2*(1-sigmoid(y-b2))

def c2_out(y):
    return q2

def two_tank_system(t,x,u):
    x1, x2 = x
    p, v = u
    x1= np.maximum(x1,0)
    x2 = np.maximum(x2,0)
    dx1dt = c1_in(x1)*(1-v)*p-c1_out(x2)*np.sqrt(x1)
    dx2dt = c2_in(x2)*v*p +c1_out(x2)*np.sqrt(x1)-q2*np.sqrt(x2)
    return np.hstack([dx1dt,dx2dt])

p_vals = np.linspace(0,1,101)
v_vals = np.linspace(0,1,101)



def main(args):
    device = 'cpu'
    set_global_seed(args.seed,deterministic=True)

    num_x0s = 21
    x0s = np.linspace(0,1,num_x0s)

    t_max = args.t_max
    n_colloc = 301

    plotting_rate = 1
    cutoff = None

    p_train = p_vals[10:-10:args.trial_rate]
    v_train = v_vals[10:-10:args.trial_rate]

    x_trials = [] 
    t_trials = []
    u_trials = []
    for pi in tqdm(p_train[:cutoff:plotting_rate]):
        for vi in v_train[:cutoff:plotting_rate]:
            
            for x0 in zip(x0s, x0s):
                u = np.array([pi,vi])

                sol = sp_solve_ivp(
                    two_tank_system,
                    t_span = [0,t_max],
                    y0 = np.array(x0),
                    t_eval= np.linspace(0,t_max, n_colloc),
                    args =(u,)
                )

                x_trials.append(sol.y.T)
                t_trials.append(sol.t.reshape(-1,1))
                u_trials.append((pi,vi))

    shuffle = True


    x_trials_tensor = [
        torch.tensor(xi,dtype=torch.float32,device=device) for xi in x_trials
    ]

    t_trials_tensor = [
        torch.tensor(ti, dtype=torch.float32,device=device) for ti in t_trials
    ]

    u_trials_tensor = [
        torch.tensor(ui, dtype=torch.float32,device=device) for ui in u_trials
    ]


    dataset = TrialsDataset(x_trials=x_trials_tensor, t_trials=t_trials_tensor, k_trials=u_trials_tensor)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=shuffle)

    f = FeluSigmoidMLP(dims=[2,10,10,2],lower_bound=-1, upper_bound=0)
    g = GeluSigmoidMLP(dims=[4,10,10,2],lower_bound=args.g_min, upper_bound=args.g_max)
    model = StabNODE(f,g).to(device)
    
    # =================================================================
    # Load params if provided
    # =================================================================
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
        train_loader=dataloader,
        n_epochs=args.epochs,
        device=device,
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


# ======================================================================
# model trainer
# ======================================================================

def to_device(tensors, device):
    return [t.to(device) if t.device != device else t for t in tensors]

def model_trainer(
        model: StabNODE,
        opt: torch.optim.Optimizer,
        loss_criteria: Callable,
        train_loader: torch.utils.data.DataLoader,
        n_epochs: int,
        min_improvement:float,
        patience: int,
        device:str='cpu',
        solve_method: str='tsit5', 
        save_folder: str|Path=None,
        show_progress:bool=True,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None,
        print_every: int=5,
        _precision: int = 4,
        effective_batch_size: int = 1,
        train_dyn = True,
        decay_scheduler: Optional[ExpLossTimeDecayScheduler] = None,
        decay_val:float = 0.0
)-> Tuple[StabNODE,dict]:
    """
    if decay_scheduler is given, this takes priority over decay_val.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
        
    
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
        for Xi, Ti, _, ui in train_loader:
            # Xi = Xi.to(device) # [batch, time, dim]
            # Ti = Ti.to(device)
            # ui = ui.to(device)
            Xi, Ti, ui = to_device([Xi,Ti,ui], device)

            x0i = to_device([Xi[:,0,:]],device)[0]



            control = lambda t: ui
            func = lambda t, x: model(t,x,control)

            opt.zero_grad()

            sol = torchode.solve_ivp(
                f=func,
                y0=x0i,
                t_eval=Ti.squeeze(),
                method=solve_method
            )
            stat = sol.status.to('cpu')
            stat = stat.item() if len(stat) == 1 else list(stat.numpy())
            epochs_status.append(stat)

            if decay_scheduler is not None:
                decay_val = decay_scheduler.get_alpha() 

            Xi_pred = sol.ys.squeeze()
            loss = loss_criteria(
                Xi_pred*torch.exp(-decay_val*Ti), 
                Xi*torch.exp(-decay_val*Ti)
            )

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
                print(
                    f"Epoch {epoch}: "
                    f"Loss = {epoch_loss:.{_precision}e}. "
                    f"time = {epoch_time:.{_precision}e}s. "
                    f"lr = {cur_lr:.{_precision}e}. "
                    f"alpha = {cur_alpha:.{_precision}e}"
                )
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StabNODE on mixing-tanks dynamics")

    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs (default: 1000)")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning rate (default: 1e-2)")
    parser.add_argument("--save_folder", type=str, default=None,
                        help="Folder to save model and logs (default: None)")
    parser.add_argument("--decay_val", type=float, default=0,
                        help="Decay value for scheduler (default: 0)")
    
    parser.add_argument("--batch_size",type=int,default=1,
                        help="Training batchsize")
    
    parser.add_argument("--t_max", type=float,default=100.,
                        help="max time-horizon for training trajectories")
    
    parser.add_argument("--trial_rate",type=int,default=10,
                        help="Rate of determining number of trials for p and v")
    
    parser.add_argument("--g_min",type=float,default=0.0,
                        help="lower bound for g function")
    
    parser.add_argument("--g_max",type=float,default=1.0,
                    help="upper bound for g function")
    
    parser.add_argument("--model_config", type=str,default=None,
                    help="Path to saved model config (.pt) for warm start.")

    parser.add_argument("--seed", type=int, default=1234,
                    help="Global randomness seed")

    args = parser.parse_args()
    main(args)