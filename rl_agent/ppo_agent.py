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
        # 策略网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # 价值网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self):
        # 这里forward不需要实现，因为actor和critic分开调用
        raise NotImplementedError

    def act(self, state):
        """
        根据状态选择动作并返回动作、log概率
        state: [state_dim]
        """
        action_probs = torch.softmax(self.actor(state), dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, state, action):
        """
        用于PPO更新时的评估函数，
        返回动作log概率、熵、状态价值。
        state: [batch, state_dim]
        action: [batch]
        """
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
        """
        PPO智能体初始化
        state_dim: 状态维度
        action_dim: 动作数目（离散）
        lr: 学习率
        gamma: 折扣因子
        K_epochs: 每次更新迭代次数
        eps_clip: PPO截断范围
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 用于旧策略记录
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        """
        选择动作
        state: np.array or torch.tensor, [state_dim]
        """
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action.item()

    def update(self):
        """
        使用收集到的数据执行一次PPO更新
        """
        # 将buffer中的数据转化为张量
        states = torch.stack(self.buffer.states, dim=0).to(self.device)
        actions = torch.stack(self.buffer.actions, dim=0).to(self.device)
        logprobs = torch.stack(self.buffer.logprobs, dim=0).to(self.device)
        
        # 计算折扣回报
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # 标准化回报
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 开始PPO更新
        for _ in range(self.K_epochs):
            # 评估旧策略下的动作在新策略下的概率比率
            action_logprobs, dist_entropy, state_values = self.policy.evaluate(states, actions)
            
            # 注意：logprobs是旧策略时记录的，而action_logprobs是新策略的
            ratios = torch.exp(action_logprobs - logprobs.detach())

            advantages = rewards - state_values.detach().squeeze()

            # PPO目标
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values.squeeze(), rewards) - 0.01*dist_entropy

            # 梯度更新
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 更新旧策略权重
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空buffer
        self.buffer.clear()

    def store_outcome(self, reward, done):
        """
        在与环境交互后，将得到的奖励与是否终止的信息存入buffer
        """
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)