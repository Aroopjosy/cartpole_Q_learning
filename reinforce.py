import gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Configuration parameters
gamma = 0.99
seed = 543
render = False
log_interval = 10

# Environment setup
env = gym.make('CartPole-v1')
try:
    env.reset(seed=seed)
except TypeError:
    print("Warning: `reset(seed=...)` may not be supported by this version of gym. The seed might not be set correctly.")
torch.manual_seed(seed)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def test_policy(num_episodes=10):
    """
    Test the trained policy on the CartPole-v1 environment.
    Args:
        num_episodes (int): Number of episodes to test the policy.
    """
    env = gym.make('CartPole-v1', render_mode='human')

    for i_episode in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward
        print(f"Test Episode {i_episode + 1}: Total Reward = {ep_reward}")
    env.close()

def evaluate_agent(num_episodes=10):
    """
    Evaluate the agent's performance without updating the policy.
    Args:
        num_episodes (int): Number of episodes to evaluate the agent.
    """
    total_reward = 0
    for i_episode in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward
        total_reward += ep_reward
        print(f"Evaluation Episode {i_episode + 1}: Total Reward = {ep_reward}")
    avg_reward = total_reward / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")

def main():
    running_reward = 10
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            if render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % log_interval == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')
        if running_reward > env.spec.reward_threshold:
            print(f"Solved! Running reward is now {running_reward} and "
                  f"the last episode runs to {t} time steps!")
            break

    print("\nEvaluating the trained policy...")
    evaluate_agent(num_episodes=10)
    
    print("\nTesting the trained policy...")
    test_policy(num_episodes=10)

   

if __name__ == '__main__':
    main()
