from gym import make
import numpy as np
from random import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm
from replaybuffer import ReplayBuffer

SEED = 0


# noinspection DuplicatedCode
class Agent:
    def __init__(self, env, gamma: float = 0.98, alpha: float = 1e-5, tau: float = 1e-2, hidden_size: int = 32):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dqn = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)
        )
        self.dqn_target = deepcopy(self.dqn)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=alpha)

        self.dqn.to(self.device)
        self.dqn_target.to(self.device)

    def update(self, batch):
        state, action, next_state, reward, done = batch

        state, next_state, reward = map(lambda item: torch.tensor(item).to(self.device).float(),
                                        (state, next_state, reward))
        action, done = map(lambda item: torch.tensor(item).to(self.device), (action, done))

        with torch.no_grad():
            q_target = self.dqn_target(next_state).max(dim=1)[0].view(-1)  # pred Q value for each action
            q_target[done] = 0

        q_target = reward + self.gamma * q_target
        qle = self.dqn(state).gather(1, action.unsqueeze(dim=1))  # take Q value for action

        loss = F.mse_loss(qle, q_target.unsqueeze(dim=1))
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clamping
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        self.soft_update()

    def act(self, state):
        state_ = torch.tensor(state).to(self.device).float()
        return torch.argmax(self.dqn(state_)).cpu().numpy().item()

    def reset(self):
        return self.env.reset()

    def train(self, transitions: int, eps_max: float = 0.5, eps_min: float = 0., buffer_size: int = 10000,
              batch_size: int = 128, shaping_coef: float = 300., progress_upd_step: int = None,
              start_training: int = 10000):
        history = ReplayBuffer(size=buffer_size)
        progress_upd_step = progress_upd_step if progress_upd_step else transitions // 100

        log = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "tau": self.tau,
            "shaping_coef": shaping_coef,
            "eps_max": eps_max,
            "eps_min": eps_min,
            "step": [],
            "reward_mean": [],
            "reward_std": []
        }

        state = self.reset()

        t = tqdm(range(transitions))
        for i in t:
            eps = eps_max - (eps_max - eps_min) * i / transitions
            if random() < eps:
                action = self.env.action_space.sample()
            else:
                action = self.act(state)

            next_state, reward, done, _ = self.env.step(action)
            reward += shaping_coef * (self.gamma * np.abs(next_state[1]) - np.abs(state[1]))
            done_ = next_state[0] >= 0.5

            history.add((state, action, next_state, reward, done_))

            state = self.reset() if done else next_state

            if i > start_training:
                self.update(history.sample(batch_size))

            if (i + 1) % progress_upd_step == 0:
                reward_mean, reward_std = self.evaluate_policy()

                log["step"].append(i)
                log["reward_mean"].append(reward_mean)
                log["reward_std"].append(reward_std)

                t.set_description(f"step: {i + 1} | Rmean = {reward_mean:0.4f} | Rstd = {reward_std:0.4f}")

        return log

    def soft_update(self):
        with torch.no_grad():
            for param, param_target in zip(self.dqn.parameters(), self.dqn_target.parameters()):
                param_target.data.mul_(1 - self.tau)
                param_target.data.add_(self.tau * param.data)

    def rollout(self, env, to_render: bool = False):
        done = False
        state = env.reset()
        total_reward = 0

        while not done:
            state, reward, done, _ = env.step(self.act(state))
            total_reward += reward
            if to_render:
                self.env.render()

        self.env.close()
        return total_reward

    def evaluate_policy(self, episodes: int = 5, to_render: bool = False):
        env = make("MountainCar-v0")
        env.seed(SEED)
        env.action_space.seed(SEED)

        rewards = []
        for _ in range(episodes):
            rewards.append(self.rollout(env, to_render=to_render))
        return np.mean(rewards), np.std(rewards)
