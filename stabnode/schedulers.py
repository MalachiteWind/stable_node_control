from typing import Optional

import numpy as np

class ExpLossTimeDecayScheduler:
    def __init__(
            self,
            init_alpha:float, 
            gamma:float, 
            min_alpha: float = 0.0, 
            patience:int=10,
            threshold:float=1e-4, 
            _alpha_thresh:float=1e-8):
        self.alpha = init_alpha
        self.gamma = gamma
        self.min_alpha = min_alpha
        self.patience = patience
        self.patience_count = 0
        self.threshold = threshold
        self.best_metric = np.inf
        self._alpha_thresh = _alpha_thresh
        self.step_history = [init_alpha]
        self.alpha_history = [init_alpha]

    def step(self, metric: float):
        if metric < self.best_metric -  self.threshold:
            self.best_metric = metric
            self.patience_count = 0
        else:
            self.patience_count += 1
        
        if self.patience_count >= self.patience:
            self.alpha *= self.gamma 
            self.patience_count = 0

            if self.min_alpha == 0.0 and self.alpha < self._alpha_thresh:
                self.alpha = 0.0
            else:
                self.alpha = max(self.alpha, self.min_alpha)
            if self.alpha_history[-1] != 0.0:
                self.alpha_history.append(self.alpha)

        
        self.step_history.append(self.alpha)
    
    def get_alpha(self):
        return self.alpha

