import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class MADDPGCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MADDPGCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions * self.n_agents
        if self.args.obs_last_action:
            self.input_shape += self.n_actions
        self.output_type = "q"

        spectral_func = spectral_norm
        if args.spectral_regularization:
            try:
                from pytorch_spectral_utils import spectral_regularize
                spectral_func = spectral_regularize
            except ImportError:
                print("pytorch_spectral_utils not found, using spectral_norm instead")

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        if args.critic_spectral[0]=="y":
            self.fc1 = spectral_func(self.fc1)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if args.critic_spectral[1]=="y":
            self.fc2 = spectral_func(self.fc2)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        if args.critic_spectral[2]=="y":
            self.fc3 = spectral_func(self.fc3)

    def forward(self, inputs, actions):
        inputs = th.cat((inputs, actions), dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # print(scheme["state"]["vshape"], scheme["obs"]["vshape"], self.n_agents, scheme["actions_one"])
        # whether to add the individual observation
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]
        # agent id
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape