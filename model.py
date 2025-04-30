import torch
import torch.nn as nn
import torch.nn.functional as F

class FurniturePPOAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        """
        PPO策略网络：用于家具放置的离散动作选择。
        """
        super(FurniturePPOAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        state_value = self.value_head(x)
        return action_logits, state_value

    def act(self, state):
        """
        采样动作，用于交互阶段。
        返回动作索引和其 log 概率。
        """
        action_logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, states, actions):
        """
        用于PPO更新阶段：返回log_prob, state_value, entropy。
        """
        action_logits, state_values = self.forward(states)
        dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, torch.squeeze(state_values), entropy
