
import torch
from torch import nn
import torch.nn.functional as F
from signjoey.helpers import freeze_params


# Uses 1D convolutional layers to capture local dependencies in sequences
class HeadNetwork(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(HeadNetwork, self).__init__()
        
        self.tc_block = nn.Sequential(
            # extract temporal dependencies
            nn.Conv1d(in_channels = in_channels, 
                                              out_channels = 1024, 
                                              kernel_size = 3, 
                                              padding = 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(in_channels = 1024, 
                                              out_channels = 1024, 
                                              kernel_size = 3, 
                                              padding = 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(in_channels = 1024, 
                                              out_channels = 1024, 
                                              kernel_size = 3, 
                                              padding = 1),                                
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        # Can be extended with a linear layer (ln) for additional feature transformation
        # self.ln = nn.Linear(1024,out_channels)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(2,1) # Swap sequence length and feature dimensions
        y = self.tc_block(x) # temporal convolutional layers
        y = y.transpose(2,1) # Restore original dimensions
        # y = self.relu(self.ln(y))
        return y

# Processes high-dimensional inputs using a deep MLP
class MultDimConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.ln_block = nn.Sequential(
            # reduce feature dimensions
            nn.Linear(in_channels,4096),
            nn.ReLU(),
            nn.Linear(4096,2048),
            nn.ReLU(),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024,out_channels),
            # normalizes outputs between -1 and 1
            nn.Softsign()
        )

    # applies the MLP layers
    def forward(self,x):
        y = self.ln_block(x)
        return y

# Deeper version of MultDimConv
class LinearHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.ln_block = nn.Sequential(
            nn.Linear(in_channels,2048),
            nn.ReLU(),
            nn.Linear(2048,4096),
            nn.ReLU(),
            nn.Linear(4096,2048),
            nn.ReLU(),
            nn.Linear(2048,out_channels),
            nn.Softsign()
        )

    def forward(self,x):
        y = self.ln_block(x)
        return y

# Shallower version of LinearHead
# Same as MultDimConv?
class LinearLink(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.ln_block = nn.Sequential(
            nn.Linear(in_channels,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,out_channels),
            nn.Softsign()
        )

    def forward(self,x):
        y = self.ln_block(x)
        return y
