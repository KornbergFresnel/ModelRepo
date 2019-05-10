import random

from collections import namedtuple


Transition = namedtuple("Transition", "state, action, next_state, reward, done")


class Buffer:
    def __init__(self, capacity):
        self._data = []
        self._capacity = capacity
        self._flag = 0

    def __len__(self):
        return len(self._data)

    def push(self, *args):
        """args: state, action, next_state, reward, done"""

        if len(self._data) < self._capacity:
            self._data.append(None)

        self._data[self._flag] = Transition(*args)
        self._flag = (self._flag + 1) % self._capacity

    def sample(self, batch_size):
        if len(self._data) < batch_size:
            return None

        samples = random.sample(self._data, batch_size)

        return Transition(*zip(*samples))
