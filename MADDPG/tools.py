import numpy as np


class Meta(object):
    obs = None
    action = None
    terminate = None
    obs_next = None
    reward = None


class Record(object):
    def __init__(self, n):
        self.loss = [0. for _ in range(n)]
        self.reward = [0. for _ in range(n)]


class Matrix(object):
    def __init__(self):
        self.data = []  # dims: (id, memory_size, inner_shape)
        self.dims = []

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.data[item[1]][item[0]]  # item: (memory, id) -> (id, memory) -> (item[1], item[0])
        else:
            return [self.data[i][item] for i in range(len(self.data))]  # item at here is memory index

    def __setitem__(self, key, value):
        if isinstance(key, int):
            for i, v in enumerate(value):
                self.data[i][key] = v
        else:
            raise Exception("Cannot accept this index type:", type(key), "You can make key as int.")

    @property
    def shape(self):
        return self.dims[0]


class ObsMatrix(Matrix):
    def __init__(self, size, obs_spaces, dtype=np.float32):
        super().__init__()
        self.dims = [(size,) + obs_spaces[i].shape for i in range(len(obs_spaces))]

        for i in range(len(obs_spaces)):
            self.data.append(np.zeros((size,) + obs_spaces[i].shape, dtype=dtype))


class ActionMatrix(Matrix):
    def __init__(self, size, action_spaces, dtype=np.float32):
        super().__init__()
        self.dims = [(size, action_spaces[i].n) for i in range(len(action_spaces))]

        for i in range(len(action_spaces)):
            self.data.append(np.zeros((size, action_spaces[i].n), dtype=dtype))


class ReplayBuffer(object):
    def __init__(self, env, config):
        # for observation and action can accept different shapes, so redesign the data structure of them
        self.obs = ObsMatrix(config.memory_size, env.observation_space)
        self.action = ActionMatrix(config.memory_size, env.action_space)
        self.reward = np.empty((config.memory_size, env.n,), dtype=np.float32)
        self.terminate = np.empty((config.memory_size, env.n,), dtype=np.bool)

        self._flag = 0  # insert position
        self._size = 0
        self._capacity = config.memory_size
        self._batch_size = config.batch_size

        assert self._capacity > 0
    
    def __len__(self):
        return self._size

    @property
    def batch_size(self):
        return self._batch_size

    def push(self, obs_set, action_set, reward_set, terminate_set):
        self.obs[self._flag] = obs_set
        self.action[self._flag] = action_set
        self.reward[self._flag] = np.array(reward_set)
        self.terminate[self._flag] = np.array(terminate_set)

        self._flag = (self._flag + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
    
    def sample(self):
        """Return a Meta object"""
        idx = np.random.choice(self._size, self._batch_size)
        data = Meta()

        if self._size < 1:
            raise Exception("You cannot sample data from a empty ReplayBuffer !")

        data.obs = self.obs[idx]
        data.obs_next = self.obs[(idx + 1) % self._size]
        data.action = self.action[idx]

        data.reward = self.reward[idx]
        data.terminate = self.terminate[idx]

        return data

    def sample_single(self, agent_id):
        """Return a Meta object, sample single agent data with agent's id"""
        idx = np.random.choice(self._size, self._batch_size)
        data = Meta()

        if self._size < 1:
            raise Exception("You cannot sample data from a empty ReplayBuffer!")
        
        if agent_id < 0 or agent_id > self.obs.shape[0]:
            raise Exception("Illegal agent id: {}".format(agent_id))
        
        data.obs = self.obs[idx, agent_id]
        data.action = self.action[idx, agent_id]

        return data

    def clear(self):
        self._flag = 0
        self._size = 0
