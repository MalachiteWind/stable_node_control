from typing import Optional

import numpy as np

class ExpLossTimeDecayScheduler:
    def __init__(
            self,
            init_alpha:float, 
            gamma:float, 
            min_alpha: float = 0.0, 
            patience:int=10,
            rtol:float=1e-4, 
            _alpha_atol:float=1e-8,
            cooldown:int=0):
        self.alpha = init_alpha
        self.gamma = gamma
        self.min_alpha = min_alpha
        self.patience = patience
        self.rtol = rtol
        self._alpha_atol = _alpha_atol
        self.cooldown = cooldown

        self.patience_count = 0
        self.best_metric = np.inf
        self.step_history = [init_alpha]
        self.alpha_history = [init_alpha]
        self.cooldown_counter = 0

    def step(self, metric: float):
        if metric < self.best_metric*(1-  self.rtol):
            self.best_metric = metric
            self.patience_count = 0
        else:
            self.patience_count += 1
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        elif self.patience_count >= self.patience:
            self.alpha *= self.gamma 
            self.patience_count = 0
            self.cooldown_counter = self.cooldown

            if self.alpha < self.min_alpha + self._alpha_atol:
                self.alpha = self.min_alpha
            # else:
            #     self.alpha = max(self.alpha, self.min_alpha)

            if self.alpha_history[-1] != self.min_alpha:
                self.alpha_history.append(self.alpha)

        
        self.step_history.append(self.alpha)
    
    def get_alpha(self):
        return self.alpha

