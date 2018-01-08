import inspect
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def extract_attrs(config):
    """This method can extract all non-inner attrs (start with `__`) from raw config.
    """

    return {k: v for k, v in inspect.getmembers(config)
            if not k.startwith("__") and not callable(k)}


class MetaData(object):
    obs = None
    action = None
    reward = None
    obs_next = None
    done = None


class ReplayBuffer(object):
    def __init__(self, batch_size, memory_size, obs_shape):
        self.flag = 0
        self.size = 0

        self.memory_size = memory_size
        self.batch_size = batch_size

        self.obs = np.empty(shape=(memory_size,) + obs_shape, dtype=np.float32)
        self.action = np.empty(shape=(memory_size,), dtype=np.int32)
        self.reward = np.empty(shape=(memory_size,), dtype=np.float32)
        self.done = np.empty(shape=(memory_size,), dtype=np.bool)

    def put(self, obs, action, reward, obs_next, done):
        self.flag = (self.flag + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)

        self.obs[self.flag, :] = obs
        self.action[self.flag] = action
        self.reward[self.flag] = reward
        self.done[self.flag] = done

        self.flag = (self.flag + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)

        # 无处安放的obs_next啊，你就现在这里暂时歇息，以后再说
        self.obs[self.flag, :] = obs_next

    def free_memory(self):
        self.flag = 0
        self.size = 0

    def sample(self):
        """Random sample batch data from inner data
        """
        batch_data = MetaData()

        idx = np.random.choice(self.size, self.batch_size)

        batch_data.obs = self.obs[idx]
        batch_data.action = self.action[idx]
        batch_data.reward = self.reward[idx]
        batch_data.obs_next = self.obs[(idx + 1) % self.memory_size]
        batch_data.done = self.done[idx]

        return batch_data


class BaseModel(object):
    def __init__(self, config):
        """Initialize configuration for DNN model"""
        self._saver = None
        self.config = config
        self.sess = None
        self.loss_record = []
        self.reward_record = []

        self.use_double = config.use_double
        self.dueling = config.use_dueling

        self.update_every = config.update_every
        self.eps_low = config.eps_low
        self.eps_high = config.eps_high
        self.eps_range = config.eps_high - config.eps_low
        self.eps_count = config.eps_count

        self.train_step = 0
        self.start_step = 0
        self.iteration = config.iteration

        self.max_to_keep = config.max_to_keep

        self.data_format = config.data_format
        self.learning_rate = config.learning_rate

        self.memory_size = config.memory_size

        # === parse config as inner attributes
        #try:
        #    self._attrs = config.__dict__['__flag']
        #except Exception as e:
        #    self._attrs = extract_attrs(config)

        #for attr in self._attrs:
        #    name = attr if not attr.startwith("_") else attr[1:]
        #    setattr(self, name, getattr(self.config, attr))

    @property
    def saver(self):
        # at most 10 recent checkpoints will be stored 
        self._saver = tf.train.Saver(max_to_keep=10) if self._saver is None else self._saver

    @property
    def model_dir(self):
        """Model dir path accept configuration from `self.config`"""
        return self.config.env_name + '/'

    @property
    def checkpoint_dir(self):
        return os.path.join("checkpoints", self.model_dir)

    def _construct_nn(self, **kwargs):
        pass

    def predict(self):
        pass

    def train(self):
        pass

    def save(self, step=None):
        """Save model to local storage, accept `step` for deciding a model file name.
        
        :param step: int, decides the file's name which stores the model
        """
        
        print("[*] Save checkpoints...")

        model_name = type(self).__name__
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.saver.save(self.sess, model_name, global_step=step)

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
