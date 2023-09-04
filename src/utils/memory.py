from collections import namedtuple
import random

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class Memory:
    """
    A simple memory buffer for storing transitions with sampling capabilities.
    """

    def __init__(self):
        self.memory = []

    def push(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def remove(self, idx):
        del self.memory[idx]

    def clear(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)
