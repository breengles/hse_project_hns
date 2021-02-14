from actor import Actor
from critic import Critic
from replaybuffer import ReplayBuffer
from gym import make
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from copy import deepcopy
from tqdm import tqdm

SEED = 0


def grad_clamp(model):
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)


class Agent:
    def __init__(self, env, alpha: float = 1e-3, gamma: float = 0.99, hidden_size: int = 32, tau: float = 1e-3):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.actor = Actor(2, hidden_size, 1)
        self.actor_target = deepcopy(self.actor)

        self.critic = Critic(3, hidden_size, 1)
        self.critic_target = deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=alpha)

        self.critic.to(self.device)
        self.critic_target.to(self.device)
        self.actor.to(self.device)
        self.actor_target.to(self.device)

    def update_critic(self, batch):
        state, action, next_state, _, done = batch

        state, next_state, action = map(lambda item: torch.tensor(item).to(self.device).float(),
                                        (state, next_state, action))
        done = torch.tensor(done).to(self.device)

        with torch.no_grad():
            q_target = self.critic_target(next_state, self.actor_target(next_state))  # pred Q value for each action
            q_target[done] = 0

        loss = F.mse_loss(self.critic(state, action), q_target)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_clamp(self.critic)

        self.critic_optimizer.step()

        self.soft_update(self.critic, self.critic_target)

    def update_actor(self, batch):
        state, *_ = batch
        state = torch.tensor(state).to(self.device).float()

        loss = -torch.mean(self.critic(state, self.actor(state)))

        self.actor_optimizer.zero_grad()
        loss.backward()
        grad_clamp(self.actor)

        self.actor_optimizer.step()

        self.soft_update(self.actor, self.actor_target)

    def act(self, state):
        with torch.no_grad():
            state_ = torch.tensor(state).to(self.device).float()
            return self.actor(state_).cpu().numpy()

    def reset(self):
        return self.env.reset()

    def train(self, transitions: int, sigma_max: float = 1., sigma_min: float = 0., buffer_size: int = 10000,
              batch_size: int = 128,
              progress_upd_step: int = None, start_training: int = 1000, shaping_coef: float = 300.):
        history = ReplayBuffer(buffer_size)
        progress_upd_step = progress_upd_step if progress_upd_step else transitions // 100

        log = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "tau": self.tau,
            "shaping_coef": shaping_coef,
            "step": [],
            "reward_mean": [],
            "reward_std": []
        }

        state = self.reset()
        t = tqdm(range(transitions))
        for i in t:
            sigma = sigma_max - (sigma_max - sigma_min) * i / transitions
            action = self.act(state)
            noise = np.random.normal(scale=sigma, size=action.shape)
            action = np.clip(action + noise, -1, 1)

            next_state, reward, done, _ = self.env.step(action)
            reward += shaping_coef * (self.gamma * np.abs(next_state[1]) - np.abs(state[1]))
            done_ = next_state[0] >= 0.5

            history.add((state, action, next_state, reward, done_))

            state = self.reset() if done else next_state

            if i > start_training:
                batch = history.sample(batch_size)
                self.update_critic(batch)
                self.update_actor(batch)

            if (i + 1) % progress_upd_step == 0:
                reward_mean, reward_std = self.evaluate_policy()

                log["step"].append(i)
                log["reward_mean"].append(reward_mean)
                log["reward_std"].append(reward_std)

                t.set_description(f"step: {i + 1} | Rmean = {reward_mean:0.4f} | Rstd = {reward_std:0.4f}")

        return log

    def soft_update(self, model, target):
        with torch.no_grad():
            for param, param_target in zip(model.parameters(), target.parameters()):
                param_target.data.mul_(1 - self.tau)
                param_target.data.add_(self.tau * param.data)

    def rollout(self, to_render: bool = False):
        done = False
        state = self.reset()
        total_reward = 0

        while not done:
            state, reward, done, _ = self.env.step(self.act(state))
            total_reward += reward
            if to_render:
                self.env.render()

        self.env.close()
        return total_reward

    def evaluate_policy(self, episodes: int = 5, to_render: bool = False):
        rewards = []
        for _ in range(episodes):
            rewards.append(self.rollout(to_render=to_render))
        return np.mean(rewards), np.std(rewards)
