import torch
from torch import nn

class Preprocesser(nn.Module):
    def __init__(self, olp_list) -> None:
        super().__init__()
        self.olp_list = olp_list
        self.eval() # nothing is trainable!


    def forward(self, x):
        if type(x) is not dict and "data" not in x:
            x = {"data": x}
        
        with torch.no_grad():
            for opt in self.olp_list:
                x = opt.forward(x)
            
        return x





