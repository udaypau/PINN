import torch
import torch.nn as nn

class NavierStokesPINN(nn.Module):
    def __init__(self, in_features=2, num_layers=6, num_hidden=40, out_features=3):
        super().__init__()
        layers = [nn.Linear(in_features, num_hidden), nn.SiLU()]
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(nn.SiLU())
            
        layers.append(nn.Linear(num_hidden, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, input_para):
        # If input is a list/tuple of tensors [x, y, t], concatenate to (N, in_features)
        if isinstance(input_para, (list, tuple)):
            xy = torch.cat(input_para, dim=1)
        else:
            xy = input_para
            
        # Ensure it is a Tensor before hitting nn.Linear
        return self.net(xy)