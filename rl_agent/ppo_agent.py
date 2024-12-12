# ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        action_probs = torch.softmax(self.actor(state), dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, state, action):

        action_probs = torch.softmax(self.actor(state), dim=-1)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, dist_entropy, state_values


class PPOAgent:
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 lr=3e-4, 
                 gamma=0.99, 
                 K_epochs=4, 
                 eps_clip=0.2, 
                 hidden_dim=64,
                 device='cpu'):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)


        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):

        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action.item()

    def update(self):


        states = torch.stack(self.buffer.states, dim=0).to(self.device)
        actions = torch.stack(self.buffer.actions, dim=0).to(self.device)
        logprobs = torch.stack(self.buffer.logprobs, dim=0).to(self.device)
        

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        for _ in range(self.K_epochs):

            action_logprobs, dist_entropy, state_values = self.policy.evaluate(states, actions)
            
            ratios = torch.exp(action_logprobs - logprobs.detach())

            advantages = rewards - state_values.detach().squeeze()


            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values.squeeze(), rewards) - 0.01*dist_entropy


            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


        self.policy_old.load_state_dict(self.policy.state_dict())


        self.buffer.clear()

    def store_outcome(self, reward, done):

        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)