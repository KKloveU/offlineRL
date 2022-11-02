import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
sys.path.append("..") 
from logger import logger
from logger import create_stats_ordered_dict
import copy
from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import CosineAnnealingLR
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x.squeeze(dim=self.dim)

class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)
    def forward(self, state):
        return self.v(state)


class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)


    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            return dist.mean if deterministic else dist.sample()

class IQL(object):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, discount=0.99, max_steps=1000000,
                 beta=3.0, EXP_ADV_MAX=100., alpha=0.005, tau=0.7):

        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = TwinQ(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.value = ValueFunction(state_dim, hidden_dim).to(device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters())

        self.max_action = max_action
        self.action_dim = action_dim

        self.policy_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.discount = discount
        self.beta = beta
        self.EXP_ADV_MAX = EXP_ADV_MAX
        self.alpha = alpha
        self.tau = tau


    def policy_loss_(self, state, perturbed_actions, y=None):
        # Update through DPG
        actor_loss = self.critic.q1(state, perturbed_actions).mean()
        return actor_loss



    def select_action(self, obs, deterministic=True):
        with torch.no_grad():
            obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
            dist = self.actor(obs)
        return dist.mean.cpu().detach().numpy().flatten() if deterministic else dist.sample().cpu().detach().numpy().flatten()


    def update_exponential_moving_average(self, target, source, alpha):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)

    def asymmetric_l2_loss(self, u, tau):
        return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


    def train(self, replay_buffer, iterations, batch_size=256):
        for it in range(iterations):
            state_np, next_state_np, action, reward, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state_np).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state_np).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(1 - done).to(device)

            with torch.no_grad():
                target_q = torch.min(*self.critic_target(state, action))
                next_v = self.value(next_state).reshape(-1,1)

            # Update value function
            v = self.value(state)
            adv = target_q - v
            v_loss = self.asymmetric_l2_loss(adv, self.tau)
            self.value_optimizer.zero_grad(set_to_none=True)
            v_loss.backward()
            self.value_optimizer.step()

            # Update Q function
            true_Q = reward + done * self.discount * next_v.detach()
            current_Q1, current_Q2 = self.critic(state, action)
            q_loss = (F.mse_loss(current_Q1, true_Q.flatten()) + F.mse_loss(current_Q2, true_Q.flatten()))/2
            self.critic_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            self.critic_optimizer.step()

            # Update policy
            exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=self.EXP_ADV_MAX)
            policy_out = self.actor(state)
            bc_losses = -policy_out.log_prob(action)
            actor_loss = torch.mean(exp_adv * bc_losses)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.policy_lr_schedule.step()

            self.update_exponential_moving_average(self.actor_target, self.actor, self.alpha)
            self.update_exponential_moving_average(self.critic_target, self.critic, self.alpha)


        logger.record_tabular('Train/Value Loss', v_loss.cpu().data.numpy())
        logger.record_tabular('Train/Actor Loss', actor_loss.cpu().data.numpy())
        logger.record_tabular('Train/Critic Loss', q_loss.cpu().data.numpy())


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.value.state_dict(), '%s/%s_value.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.value.load_state_dict(torch.load('%s/%s_value.pth' % (directory, filename)))
