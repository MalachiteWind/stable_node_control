import time
import pickle
import copy
import warnings

import numpy as np
import torch 
import torch.nn as nn
from torch.optim import Optimizer
from torchdiffeq import odeint
from torchode import solve_ivp


from pathlib import Path
from typing import Optional, Callable, Tuple
from .utils import _load_loop_wrapper



class MLP(torch.nn.Module):
    def __init__(self, dims, activation=torch.nn.SiLU(), dtype=torch.float):
        # Base constructor
        super().__init__()

        # Store values
        self.dims = list(dims)
        self.activation = activation
        self.dtype = dtype

        # Create layers
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, output_dim).to(self.dtype)
            for input_dim, output_dim in zip(self.dims[:-1], self.dims[1:])
        ])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x




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

class GeluSigmoid(nn.Module):
    def __init__(
        self,
        dim_in, 
        dim_out, 
        hidden_dim = 2,
        lower_bound=0,
        upper_bound=1
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim_out),
        )

        self.args = {
            "dim_in": dim_in, 
            "dim_out": dim_out, 
            "hidden_dim": hidden_dim,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

    def forward(self,x,u):
        xu = torch.cat([x,u],dim=-1)
        a = self.args['lower_bound']
        b = self.args['upper_bound']

        return a + (b-a)*torch.sigmoid(self.network(xu))


class FeluSigmoid(nn.Module):
    def __init__(
            self,
            dim_in, 
            dim_out, 
            hidden_dim = 2,
            lower_bound=0,
            upper_bound=1
        ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim_out),
        )

        self.args = {
            "dim_in": dim_in, 
            "dim_out": dim_out, 
            "hidden_dim": hidden_dim,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
        
    def forward(self,x):
        a = self.args["lower_bound"]
        b = self.args["upper_bound"]
        return a + (b-a)*torch.sigmoid(self.network(x))
    

class FeluSigmoidMLP(nn.Module):
    def __init__(
            self, 
            dims, 
            activation=torch.nn.SiLU(),
            lower_bound=0,
            upper_bound=1
        ):
        super().__init__()
        self.dims = dims
        self.activation = activation
        self.network = MLP(self.dims, activation = self.activation)

        self.args = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
        
    def forward(self,x):
        a = self.args["lower_bound"]
        b = self.args["upper_bound"]
        return a + (b-a)*torch.sigmoid(self.network(x))
    



class GeluSigmoidMLP(nn.Module):
    def __init__(
        self,
        dims,
        activation = torch.nn.SiLU(),
        lower_bound=0,
        upper_bound=1
    ):
        super().__init__()

        self.dims = dims
        self.activation = activation
        self.network = MLP(self.dims, activation = self.activation)

        self.args = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

    def forward(self,x,u):
        xu = torch.cat([x,u],dim=-1)
        a = self.args['lower_bound']
        b = self.args['upper_bound']

        return a + (b-a)*torch.sigmoid(self.network(xu))
    


class GeluSigmoidMLPfeaturized(nn.Module):
    def __init__(
        self,
        dims,
        activation = torch.nn.SiLU(),
        lower_bound=0,
        upper_bound=1,
        freq_sample_step = 5
    ):
        super().__init__()

        self.dims = dims
        self.activation = activation
        self.network = MLP(self.dims, activation = self.activation)

        self.freq_sample_step = freq_sample_step
        self.featurization_dim = dims[0] - 2
        self.freqs = torch.arange(0,self.featurization_dim*self.freq_sample_step,self.freq_sample_step)
        

        self.args = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

    def forward(self,x,u):
        a = self.args['lower_bound']
        b = self.args['upper_bound']
        x_feats = [x]
        for fq in self.freqs:
            x_feats.append(torch.cos(fq**2*3.14*(x- a)/(b-a)))
        xf = torch.cat(x_feats,dim=-1)
        xu = torch.cat([xf,u],dim=-1)


        return a + (b-a)*torch.sigmoid(self.network(xu))





class StabNODE(nn.Module):
    def __init__(self,f:Felu,g:Gelu):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, t, x, u_func):
        """
        t: scalar or shape [B]
        x: shape [B, d]
        """
        u = u_func(t)  # shape [B, d_u]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, d]
        if u.dim() == 1:
            u = u.unsqueeze(0)  # [1, d_u]
        fx = self.f(x)          # shape [B, d]
        gx = self.g(x, u)       # shape [B, d]
        return fx * (x - gx)    # shape [B, d]


def model_trainer(
        model: StabNODE,
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
        exp_loss_time_decay = 0
)-> Tuple[StabNODE,dict]:
    
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
    model.train()
    max_iters = len(train_loader)
     #this is training iteration counter to keep track of effective batch size.
    for epoch in loop_wrapper(range(n_epochs)):
        t1 = time.time()
        epoch_loss = 0.0
        num_batches = 0
        epochs_status = []
        iter_counter = 0
        for Xi, Ti, x0i in train_loader:
            Xi = Xi.squeeze() # [batch, time, dim]
            Ti = Ti.squeeze()
            x0i = x0i.reshape(-1,1)

            if not x0i.requires_grad:
                x0i = x0i.clone().detach().requires_grad_()


            opt.zero_grad()


            if train_dyn == True:
                sol = solve_ivp(
                    f=lambda t, x: model(t, x, control),
                    y0=x0i,
                    t_eval=Ti,
                    method=solve_method
                )

                epochs_status.append(sol.status)
                Xi_pred = sol.ys.squeeze()
                loss = loss_criteria(Xi_pred*torch.exp(-exp_loss_time_decay*Ti), Xi*torch.exp(-exp_loss_time_decay*Ti))

            Xi = Xi.unsqueeze(-1)
            cntrl = control(Ti)
            cntrl = torch.reshape(cntrl,(1,1))
            cntrl = cntrl.repeat(Xi.shape[0],1)
            g_id_loss = 100*loss_criteria(model.g(Xi,cntrl),Xi)

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
        if scheduler is not None:
            scheduler.step(epoch_loss)

        cur_lr = opt.param_groups[0]['lr']
        epoch_time = time.time() - t1

        losses.append(epoch_loss)
        times.append(epoch_time)
        status.append(epochs_status)
        lr_hist.append(cur_lr)

        if show_progress:
            if epoch <= 5 or epoch % print_every == 0:
                print(f"Epoch {epoch}: Loss: {epoch_loss:.{_precision}f}. time = {epoch_time:.{_precision}f}s. lr = {cur_lr:.{_precision}f}")    
        
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


def _save_model_opt_cpu(model:StabNODE, opt, epoch, loss, save_path:str, scheduler = None):
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
    

    checkpoint = {
        "f_state_dict": f_state,
        "g_state_dict": g_state,
        "stabnode_state_dict": model_state,
        "f_args": f_args,
        "g_args":g_args,
        "opt_state_dict": opt.state_dict(),
        "epoch": epoch,
        "loss": loss,}

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    

    torch.save(checkpoint,save_path)
    
def _save_log_history(
        losses,
        times,
        stopping_criteria,
        best_model_epoch,
        method_status,
        patience_hist,
        lr_hist,
        save_path:str = None,
):
    log_history = {
        "losses": losses,
        "times": times,
        "stopping_criteria": stopping_criteria,
        "best_model_epoch": best_model_epoch,
        "method_status": method_status,
        "patience_hist": patience_hist,
        "lr_hist": lr_hist
    }

    if save_path is not None:
        with open(save_path, 'wb') as f: 
            pickle.dump(log_history, f)
    
    return log_history


def _create_save_paths(folder: str | Path):
    if folder is None:
        return None, None
    
    base_path = Path(folder)
    base_path.mkdir(parents=True, exist_ok=True)  

    
    model_opt_path = base_path / "model_opt_states.pt"
    log_path = base_path / "log_hist.pkl"

    return str(model_opt_path), str(log_path)



def _load_model_opt(save_path:str, device:str = 'cpu'):
    config = torch.load(save_path, map_location='cpu',weights_only=False)

    f = Felu(**config["f_args"])
    g = Gelu(**config["g_args"])
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
