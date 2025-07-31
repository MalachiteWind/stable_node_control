import time
import pickle
import copy
import warnings

import numpy as np
import torch 
import torch.nn as nn
from torch.optim import Optimizer
from torchdiffeq import odeint


from pathlib import Path
from typing import Optional, Callable, Tuple
from .utils import _load_loop_wrapper

def set_global_seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FTerm(nn.Module):
    def __init__(self,dim_in, dim_out, hidden_dim = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim_out),
            nn.Tanh(),
        )
        self.args = {
            "dim_in": dim_in, 
            "dim_out": dim_out, 
            "hidden_dim": hidden_dim
        }
        
    def forward(self,x):
        return - torch.exp(self.network(x))


class Gelu(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dim = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim_out),
            nn.Tanh(),
        )
        self.args = {
            "dim_in": dim_in,
            "dim_out": dim_out,
            "hidden_dim": hidden_dim
        }

    def forward(self, x, u):
        xu = torch.cat([x,u],dim=-1)
        return self.network(xu)


class Felu(nn.Module):
    def __init__(self,dim_in, dim_out, hidden_dim = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim_out),
        )
        self.args = {
            "dim_in": dim_in, 
            "dim_out": dim_out, 
            "hidden_dim": hidden_dim
        }
        
    def forward(self,x):
        return - torch.exp(self.network(x))


class GTerm(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dim = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim_out),
            nn.Tanh(),
        )
        self.args = {
            "dim_in": dim_in,
            "dim_out": dim_out,
            "hidden_dim": hidden_dim
        }

    def forward(self, x, u):
        xu = torch.cat([x,u],dim=-1)
        return self.network(xu)


class StabNODE(nn.Module):
    def __init__(self,f:FTerm,g:GTerm):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, t, state, u_func):
        x = state
        u = u_func(t).unsqueeze(0)
        fx = self.f(x)
        gx = self.g(x,u)
        return fx*(x-gx)


def _save_model_opt_cpu(model:StabNODE, opt, epoch, loss, save_path:str):
    device = next(model.parameters()).device.type
    f = model.f
    g = model.g
    f_args = f.args
    g_args = g.args

    if device  == "cpu":
        f_state = f.state_dict()
        g_state = g.state_dict()
        model_state = model.state_dict()
    else:
        f_cpu = copy.deepcopy(f).to('cpu')
        g_cpu = copy.deepcopy(g).to('cpu')
        model_cpu = StabNODE(f_cpu,g_cpu).to('cpu')

        f_state = f_cpu.state_dict()
        g_state = g_cpu.state_dict()
        model_state = model_cpu.state_dict()

    torch.save({
        "f_state_dict": f_state,
        "g_state_dict": g_state,
        "stabnode_state_dict": model_state,
        "f_args": f_args,
        "g_args":g_args,
        "opt_state_dict": opt.state_dict(),
        "epoch": epoch,
        "loss": loss},
        save_path)

def _load_model_opt(save_path:str, device:str = 'cpu'):
    config = torch.load(save_path, map_location='cpu',weights_only=False)

    f = FTerm(**config["f_args"])
    g = GTerm(**config["g_args"])
    model = StabNODE(f,g)

    f.load_state_dict(config["f_state_dict"])
    g.load_state_dict(config["g_state_dict"])
    model.load_state_dict(config["stabnode_state_dict"])
    model.to(device)

    opt = torch.optim.Adam(model.parameters())
    opt.load_state_dict(config["opt_state_dict"])

    epoch = config["epoch"]
    loss = config["loss"]

    return model, opt, epoch, loss

    

def _save_log_history(
        losses,
        times,
        stopping_criteria,
        best_model_epoch,
        method_failures,
        patience_hist,
        save_path:str = None,
):
    log_history = {
        "losses": losses,
        "times": times,
        "stopping_criteria": stopping_criteria,
        "best_model_epoch": best_model_epoch,
        "method_failures": method_failures,
        "patience_hist": patience_hist,
    }

    if save_path is not None:
        with open(save_path, 'wb') as f: 
            pickle.dump(log_history, f)
    
    return log_history

def _create_save_paths(path):
    if path is None:
        return None, None
    base_path = Path(path)
    folder = base_path.parent

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    
    model_path = base_path
    log_path = folder / (base_path.stem+ "_log.pkl")
    return str(model_path), str(log_path)


def model_trainer(
        model: StabNODE,
        opt: torch.optim.Optimizer, 
        loss_criteria:Callable, 
        x0:torch.Tensor,
        tau_span:torch.Tensor, 
        X_train:torch.Tensor,
        control: Callable[[torch.Tensor], torch.Tensor], 
        n_epochs: int, 
        min_improvement: float, 
        patience: int, 
        print_every: int,
        solve_method: str = "rk4",
        show_progress: bool = True,
        save_path: Optional[str] = None
) -> Tuple[StabNODE,dict]:

    loop_wrapper = _load_loop_wrapper(show_progress)
    model_opt_save_path, log_save_path = _create_save_paths(save_path)

    best_loss = torch.inf
    patience_counter = 0 
    stopping_criteria = "max-epochs"
    best_model_epoch = -1



    losses = []
    times = []
    method_failures = []
    patience_hist = []
    model.train()
    for epoch in loop_wrapper(range(n_epochs)):
        t1 = time.time()
        opt.zero_grad()

        # If solve_method causes numerical blowup,
        # try more robust default method "dopri15".

        # forward pass
        try:
            X_pred = odeint(
                lambda t, x: model(t,x,control),
                x0, 
                tau_span, 
                method=solve_method
            )

            if torch.isnan(X_pred).any():
                raise ValueError("NaN in ODE Solution.")
            method_failures.append(False) 
            
        except ValueError as e:
            warnings.warn(f"{solve_method} failure. Using fallback method 'dopri5'. Error: {e}")
            X_pred = odeint(lambda t, x: model(t,x,control), x0, tau_span, method='dopri5')
            method_failures.append(True)
        
        loss = loss_criteria(X_pred.squeeze(), X_train.squeeze())

        # backwards pass
        loss.backward()
        opt.step()

        t2 = time.time()

        # Record diagnostics
        epoch_time = t2 - t1
        epoch_loss = loss.item()

        # patience_counter
        # stopping_critiera: early-stoppage, max-iteration

        losses.append(epoch_loss)
        times.append(epoch_time)

        if show_progress:
            if epoch <= 5 or epoch % print_every == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, time = {epoch_time:.4f}")
        
        # Check early stop criteria 
        if best_loss - epoch_loss >= min_improvement:
            best_loss = epoch_loss
            patience_counter = 0
            best_model_epoch = epoch
            if save_path is not None:
                _save_model_opt_cpu(model,opt,best_model_epoch,best_loss,model_opt_save_path)
        else:
            patience_counter += 1
        
        patience_hist.append(patience_counter)

        if patience_counter > patience:
            stopping_criteria = "early-stoppage"
            if show_progress is not None:
                print(f"Patience exceeded: {patience} . Early stoppage executed.")
            break
        if save_path is not None:
            _=_save_log_history(
                losses,
                times,
                f"checkpoint-{epoch}",
                best_model_epoch,
                method_failures,
                patience_hist,
                log_save_path
            )
        
    log_history = _save_log_history(
        losses, 
        times, 
        stopping_criteria, 
        best_model_epoch, 
        method_failures,
        patience_hist,
        log_save_path
    )

    return model, log_history

