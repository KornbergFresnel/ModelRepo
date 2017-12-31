import numpy as np


class History(object):
    def __init__(self, config):
        """History implements the frames stacking, its inner data-structure
        is related with data format (`NCHW` or `NHWC`)
        """

        self.cnn_format = config.cnn_format
        obs_height, obs_width, history_length = \
                config.obs_height, config.obs_width, config.history_length

        if self.cnn_format == "NCHW":
            self._history = np.zeros(shape=(history_length, obs_height, obs_width), dtype=np.float32)
        else:
            self._history = np.zeros(shape=(obs_height, obs_width, history_length), dtype=np.float32)

    def add(self, obs_frame):
        """Accept an screen frame then append it to the tail while pop the head frame
        """
        # replace the former `history_length - 1` with the last `history_length - 1` frames
        self._history[:-1] = self._history[1:]
        self._history[-1] = obs_frame

    def get(self):
        return self._history


class ReplayBuffer(object):
    def __init__(self, config):
        """ReplayBuffer work for off-policy, it will store experiences for agent,
        and its inner data-structure will store some `(s_t, action_t, reward_t, terminal)` sequences
        """

        self.memory_size, obs_height, obs_width = config.memory_size, config.obs_height, config.obs_width

        self._obs_mem = np.empty(shape=(self.memory_size, obs_height, obs_width), dtype=np.float32)
        self._action_mem = np.empty(shape=(self.memory_size, 1), dtype=np.uint8)
        self._reward_mem = np.empty(shape=(self.memory_size, 1), dtype=np.float32)
        self._terminal_mem = np.empty(shape=(self.memory_size, 1), dtype=np.bool)

        self.pos_flag = 0  # indicate the position which newest sequence will insert
        self.dims = (config.obs_height, config.obs_width)   # matains a shape of coming observation
        self.counter = 0  # matains a counter for experiences storage

        self.history_length = config.history_length
        self.batch_size = config.batch_size

    def add(self, obs_t, action_t, reward_t, terminal):
        """Will store newest experience and drop oldest experience
        """

        assert obs_t.shape == self.dims

        self._obs_mem[self.pos_flag] = obs_t
        self._action_mem[self.pos_flag] = action_t
        self._reward_mem[self.pos_flag] = reward_t
        self._terminal_mem[self.pos_flag] = terminal

        # update indication
        self.pos_flag = (self.pos_flag + 1) % self.memory_size
        self.counter += 1 if self.counter < self.memory_size else 0

    def sample(self):
        """Will sample some experiences from its storage
        """

        assert self.counter > self.history_length

        idx = np.random.choice(self.counter, self.batch_size)
        obs_batch = self._obs_mem[idx, :]
        action_batch = self._action_mem[idx, :]
        obs_next_bath = self._obs_mem[(idx + 1) % self.counter, :]
        terminal_batch = self._terminal_mem[idx, :]

        return obs_batch, action_batch, reward_batch, obs_next_bath, terminal_batch
        
