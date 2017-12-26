import random
import tensorflow as tf
import numpy as np
import pprint
import inspect
import os


def extract_attrs(config):
    """This method can extract all non-inner attrs (start with `__`) from raw config.
    """

    return {k: v for inspect.getmembers(config)
            if not k.startwith("__") and not callable(k)}

class BaseModel(object):
    def __init__(self, config):
        """Initialize configuration for DNN model"""
        self._saver = None
        self.config = config
        self.sess = None

        # === parse config as inner attributes
        try:
            self._attrs = config.__dict__['__flag']
        except:
            self._attrs = extract_attrs(config)

        for attr in self._attrs:
            name = attr if not attr.startwith("_") else attr[1:]
            setattr(self, name, getattr(self.config, attr))

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
        """Construct neural network, this module may different for different demand.
        """
        pass

    def predict(self):
        """Make prediction, this module should be a custom module for different damand.
        """
        pass

    def train(self):
        """Neural network tranining module, this module shoudl be a custom module for different demand.
        """
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

