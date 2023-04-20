import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class MLP(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_dim, args):
        super(MLP, self).__init__()

        # spectral_func = spectral_regularize if args.spectral_regularization else spectral_norm
        spectral_func = spectral_norm
        if args.spectral_regularization:
            try:
                from pytorch_spectral_utils import spectral_regularize
                spectral_func = spectral_regularize
            except ImportError:
                print("pytorch_spectral_utils not found, using spectral_norm instead")

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if args.critic_spectral[0]=="y":
            self.fc1 = spectral_func(self.fc1)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if args.critic_spectral[1]=="y":
            self.fc2 = spectral_func(self.fc2)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        if args.critic_spectral[2]=="y":
            self.fc3 = spectral_func(self.fc3)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q