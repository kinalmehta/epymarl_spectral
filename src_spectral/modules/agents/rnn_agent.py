import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        spectral_func = spectral_norm
        if args.spectral_regularization:
            try:
                from pytorch_spectral_utils import spectral_regularize
                spectral_func = spectral_regularize
            except ImportError:
                print("pytorch_spectral_utils not found, using spectral_norm instead")
        
        self.fc1 = (nn.Linear(input_shape, args.hidden_dim))
        if args.policy_spectral[0]=="y":
            self.fc1 = spectral_func(self.fc1)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
            if args.policy_spectral[1]=="y":
                self.rnn = spectral_func(self.rnn, name="weight_hh")
            if args.policy_spectral[2]=="y":
                self.rnn = spectral_func(self.rnn, name="weight_ih")
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
            if args.policy_spectral[1]=="y":
                self.rnn = spectral_func(self.rnn)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
        if args.policy_spectral[3]=="y":
            self.fc2 = spectral_func(self.fc2)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h

