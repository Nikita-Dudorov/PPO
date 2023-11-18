from torch.distributions.categorical import Categorical
from torch import nn

class ActorCritic(nn.Module):
    """Implements actor-critic agent for raw observation and discrete action space"""

    def __init__(self, n_hidden, obs_dim, act_dim):
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(n_hidden, 1, bias=True)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.Tanh(),
            nn.Linear(n_hidden, act_dim, bias=True)
        )

        # TODO add custom layer init 

    def get_value(self, obs):
        return self.critic(obs)

    def get_action(self, obs):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        act = dist.sample()
        # return action, action probability, entropy of action distribution
        return act, dist.log_prob(act).exp(), dist.entropy()