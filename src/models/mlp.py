import torch
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.dropout = dropout
        
        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index=None): # edge_index is unused
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x