import numpy as np
from random import random
from tqdm import tqdm


class Agent:
    def __init__(self, env, alpha: float = 0.1, gamma: float = 0.98, grid_size_x: int = 30, grid_size_y: int = 30):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y

        self.action_dim = 3
        self.qle = np.zeros((self.grid_size_x * self.grid_size_y, self.action_dim)) + 2.  # kinda boosting

    def act(self, state, eps: float = 0.1):
        state = self.transform_state(state)
        return np.argmax(self.qle[state])

    def update(self, transition):
        state, action, next_state, reward, done = transition
        state, next_state = self.transform_state(state), self.transform_state(next_state)

        if done:
            self.qle[next_state] = 0

        self.qle[state, action] += self.alpha * (reward
                                                 + self.gamma * np.max(self.qle[next_state]) - self.qle[state, action])

    def reset(self):
        return self.env.reset()

    def train(self, transitions=4_000_000, eps=0.1, shaping_coef=300.):
        trajectory = []

        log = {
            "eps": eps,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "shaping_coef": shaping_coef,
            "step": [],
            "mean": [],
            "std": []
        }

        state = self.reset()
        t = tqdm(range(transitions))
        for i in t:
            # eps *= (transitions + 1 - i) / transitions  # quite strong
            eps *= (1 - i / transitions)  # linear

            if random() < eps:
                action = self.env.action_space.sample()
            else:
                action = self.act(state, eps=eps)

            next_state, reward, done, _ = self.env.step(action)
            reward += shaping_coef * (
                    self.gamma * np.abs(next_state[1]) - np.abs(state[1]))  # potential reward shaping
            done_ = next_state[0] > 0.5  # check whether or not we actually achieved the goal

            trajectory.append((state, action, next_state, reward, done_))
            if done:
                for transition in reversed(trajectory):
                    self.update(transition)
                trajectory = []

            state = self.reset() if done else next_state

            if (i + 1) % (transitions // 100) == 0:
                r_mean, r_std = self.evaluate_policy(episodes=10)
                t.set_description(f"step: {i + 1} | Rmean = {r_mean:0.4f} | Rstd = {r_std:0.4f}")

                log["step"].append(i + 1)
                log["mean"].append(r_mean)
                log["std"].append(r_std)

        return log

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

    def evaluate_policy(self, episodes: int = 100, to_render: bool = False):
        rewards = []
        for _ in range(episodes):
            rewards.append(self.rollout(to_render=to_render))
        return np.mean(rewards), np.std(rewards)

    def transform_state(self, state):
        state = (np.array(state) + np.array((1.2, 0.07))) / np.array((1.8, 0.14))
        x = min(int(state[0] * self.grid_size_x), self.grid_size_x - 1)
        y = min(int(state[1] * self.grid_size_y), self.grid_size_y - 1)
        return x + self.grid_size_x * y

    def save(self, file="agent.npy"):
        np.save(file, self.qle)
