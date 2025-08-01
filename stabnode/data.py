from typing import Optional

import torch

class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, X, t, window_size:Optional[int]=None ):
        self.X = X              # shape [T_total, d]
        self.t = t              # shape [T_total]
        self.window_size = window_size

    def __len__(self):
        if self.window_size is None:
            return 1
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError(
                f"Index {idx} is out of bounds of dataset size: {self.__len__()}."
            )
        
        if self.window_size is None:
            return self.X, self.t, self.X[0]
        
        x0 = self.X[idx]                                          
        t_window = self.t[idx : idx + self.window_size]           
        x_window = self.X[idx : idx + self.window_size]           

        return x_window, t_window, x0