import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from logger import logger
from logger import create_stats_ordered_dict
import copy


def softmax(x,  axis=0):
    x = x - x.max()
    return torch.exp(x) / torch.sum(torch.exp(x), axis=axis, keepdims=True)


class Actor(nn.Module):
    """Actor used in BCQ"""
    def __init__(self, state_dim, action_dim, max_action, threshold=0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        self.threshold = threshold

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = self.threshold * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)

class Critic(nn.Module):
    """Regular critic used in off-policy RL"""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class VAE(nn.Module):
    """VAE Based behavior cloning also used in Fujimoto et.al. (ICML 2019)"""
    def __init__(self, state_dim, action_dim, latent_dim, max_action, activation='tanh'):
        super(VAE, self).__init__()

        # encoder
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)
        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        # decoder
        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.activation = activation


    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))
        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(device)
        u = self.decode(state, z)
        return u, mean, std

        
    def decode(self, state, z=None):
        if z is None:
                z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.5, 0.5)
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))


    def decode_bc_test(self, state, z=None):
        if z is None:
                z = torch.FloatTensor(np.random.normal(0, 1, size=(state.size(0), self.latent_dim))).to(device).clamp(-0.25, 0.25)
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))
    



class BCQ(object):
    def __init__(self, state_dim, action_dim, max_action, cloning=False, discount=0.99):
        latent_dim = action_dim * 2

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.max_action = max_action
        self.action_dim = action_dim
        self.use_cloning = cloning
        self.discount = discount


    def policy_loss_(self, state, perturbed_actions, y=None):

        # Update through DPG
        actor_loss = self.critic.q1(state, perturbed_actions).mean()

        return actor_loss

    def sample_action(self, state):
        sampled_actions = self.vae.decode(state)
        perturbed_actions = self.actor(state, sampled_actions)
        return perturbed_actions

    def select_action(self, state):
        if self.use_cloning:
            return self.select_action_cloning(state)       
        with torch.no_grad():
                state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1).to(device)
                action = self.actor(state, self.vae.decode(state))
                q1 = self.critic.q1(state, action)
                ind = q1.max(0)[1]
        return action[ind].cpu().data.numpy().flatten()

    def select_action_cloning(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.vae.decode_bc_test(state)
        return action[0].cpu().data.numpy().flatten()

    def compute_grad_norm(self, net):
        grad_norm = []
        for p in list(filter(lambda p: p.grad is not None, net.parameters())):
                grad_norm.append(p.grad.flatten())
        grad_norm = torch.cat(grad_norm).norm(2).sum() + 1e-12
        return grad_norm

    def train(self, replay_buffer, iterations, batch_size=100, tau=0.005):
        for it in range(iterations):
            state_np, next_state_np, action, reward, done = replay_buffer.sample(batch_size)
            state           = torch.FloatTensor(state_np).to(device)
            action          = torch.FloatTensor(action).to(device)
            next_state      = torch.FloatTensor(next_state_np).to(device)
            reward          = torch.FloatTensor(reward).to(device)
            done            = torch.FloatTensor(1 - done).to(device)

            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss
            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            # Critic Training
            with torch.no_grad():
                # Duplicate state 10 times
                state_rep = torch.FloatTensor(np.repeat(next_state_np, 10, axis=0)).to(device)
                # Compute value of perturbed actions sampled from the VAE
                if self.use_cloning:
                    target_Q1, target_Q2 = self.critic_target(state_rep, self.vae.decode(state_rep))
                else:
                    target_Q1, target_Q2 = self.critic_target(state_rep, self.actor_target(state_rep, self.vae.decode(state_rep)))
                # Soft Clipped Double Q-learning 
                target_Q = 0.75 * torch.min(target_Q1, target_Q2) + 0.25 * torch.max(target_Q1, target_Q2)
                target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)
                target_Q = reward + done * self.discount * target_Q
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


            # Pertubation Model / Action Training
            with torch.no_grad():
                sampled_actions = self.vae.decode(state)
            perturbed_actions = self.actor(state, sampled_actions)
            # Update through DPG
            actor_loss = -self.critic.q1(state, perturbed_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        logger.record_tabular('Train/VAE Loss', vae_loss.cpu().data.numpy())
        logger.record_tabular('Train/Actor Loss', actor_loss.cpu().data.numpy())
        logger.record_tabular('Train/Critic Loss', critic_loss.cpu().data.numpy())

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename)))
