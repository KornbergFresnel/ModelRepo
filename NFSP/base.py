import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class MetaData(object):
    obs = None
    action = None
    reward = None
    obs_next = None
    done = None


class ReplayBuffer(object):
    """This replay buffer is designed for data storage of Reinforcement Learning
    """

    def __init__(self, batch_size, memory_size, obs_shape):
        self._flag = 0
        self._size = 0

        self.memory_size = memory_size
        self.batch_size = batch_size

        self.obs = np.empty(shape=(memory_size,) + obs_shape, dtype=np.float32)
        self.action = np.empty(shape=(memory_size,), dtype=np.int32)
        self.reward = np.empty(shape=(memory_size,), dtype=np.float32)
        self.done = np.empty(shape=(memory_size,), dtype=np.bool)
    
    @property
    def size(self):
        return self._size

    def put(self, obs, action, reward, done):
        self.obs[self._flag, :] = obs
        self.action[self._flag] = action
        self.reward[self._flag] = reward
        self.done[self._flag] = done

        self._flag = (self._flag + 1) % self.memory_size
        self._size = min(self._size + 1, self.memory_size)

    def free_memory(self):
        self._flag = 0
        self._size = 0

    def sample(self):
        """Random sample batch data from inner data
        """
        batch_data = MetaData()

        idx = np.random.choice(self._size, self.batch_size)

        batch_data.obs = self.obs[idx]
        batch_data.action = self.action[idx]
        batch_data.reward = self.reward[idx]
        batch_data.obs_next = self.obs[(idx + 1) % self.memory_size]
        batch_data.done = self.done[idx]

        return batch_data


class SReplayBuffer(object):
    def __init__(self, batch_size, memory_size, obs_shape, action_size):
        self._size = 0
        self._flag = 0

        self.batch_size = batch_size
        self._memory_size = memory_size

        self.obs = np.empty(shape=(memory_size,) + obs_shape, dtype=np.float32)
        self.action = np.empty(shape=(memory_size, action_size), dtype=np.int32)
    
    @property
    def size(self):
        return self._size
    
    def put(self, obs, action):
        self.obs[self._flag, :] = obs
        self.action[self._flag, action] = 1

        self._flag = (self._flag + 1) % self._memory_size
        self._size = min(self._size + 1, self._memory_size)
    
    def sample(self):
        """Random sample batch-data from inner data"""
        batch_data = MetaData()

        idx = np.random.choice(self._size, self.batch_size)

        batch_data.obs = self.obs[idx]
        batch_data.action = self.action[idx]


class BaseModel(object):
    def __init__(self, name, config):
        """Initialize configuration for DNN model"""
        self._saver = None
        self.config = config

        self.name = name

        self.sess = None
        self.loss_record = []
        self.reward_record = []

        self.use_double = config.use_double
        self.dueling = config.use_dueling

        self.update_every = config.UPDATE_EVERY
        self.save_every = config.SAVE_EVERY

        self.eps = config.eps_high
        self.eps_decay = config.eps_decay

        self.train_step = 0
        self.start_step = 0
        self.iteration = config.iteration

        self.max_to_keep = config.max_to_keep

        self.data_format = config.data_format
        self.learning_rate = config.learning_rate

        self.memory_size = config.memory_size
        self.env_name = config.ENV_NAME

    @property
    def saver(self):
        # at most 10 recent checkpoints will be stored 
        self._saver = tf.train.Saver(max_to_keep=10) if self._saver is None else self._saver
        return self._saver

    @property
    def model_dir(self):
        """Model dir path accept configuration from `self.config`"""
        return self.env_name + '/'

    @property
    def checkpoint_dir(self):
        return os.path.join("checkpoints", self.model_dir)

    def _construct_nn(self, **kwargs):
        pass

    def predict(self):
        pass

    def train(self, num_train):
        pass

    def save(self, step=None):
        """Save model to local storage, accept `step` for deciding a model file name.
        
        :param step: int, decides the file's name which stores the model
        """
        
        print("[*] Save checkpoints...")

        model_name = type(self).__name__
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.saver.save(self.sess, self.checkpoint_dir + model_name, global_step=step)

    def load(self, step=None):
        """Load model from storage, this method accept a `int` parameter to indicate the certain model file.

        :param step: int, decides the file's name which stores the model
        """

        print("[*] Loading checkpoints...")

        # check the existence of checkpoint dir, return `CheckPointState` if the state is avaliable, otherwise `None`
        ckp_state = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckp_state and ckp_state.model_checkpoint_path:
            ckpt_name = os.path.basename(ckp_state)
            f_name = os.path.join(self.checkpoint_dir, ckpt_name)
            self.saver.restore(self.sess, f_name)
            print("[*] >> SUCCESS, loaded from: {}".format(f_name))
            return True
        else:
            print("[!] >> Load FAILED: {}".format(self.checkpoint_dir))
            return False

    def plot(self, *args):
        plt.plot(np.arange(len(self.reward_record)), self.reward_record, color="green", linestyle="dashed")
        plt.xlabel("Training Episode")
        plt.ylabel("Average Loss on 512 Iteration")
        plt.show()

    def record(self):
        if not os.path.exists("data"):
            os.mkdir("data")

        import pickle

        with open("data/log_loss.pkl", "wb") as f:
            pickle.dump(self.loss_record, f)

        with open("data/log_reward.pkl", "wb") as f:
            pickle.dump(self.reward_record, f)

        print("[* Saver] --Record done!--")
