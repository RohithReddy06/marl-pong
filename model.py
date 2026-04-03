import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2)):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, 0)
    return layer

class PPOAgent(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 10 features (8 physics + 2 Role IDs)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(10, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(10, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 3), std=0.01)
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)