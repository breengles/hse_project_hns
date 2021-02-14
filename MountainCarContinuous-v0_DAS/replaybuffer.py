from collections import deque
from random import sample


class ReplayBuffer:
    def __init__(self, size: int = 10000):
        self.buffer = deque(maxlen=size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, size):
        assert len(self.buffer) >= size
        tmp = sample(self.buffer, size)
        return list(zip(*tmp))

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
