import tensorflow as tf
import numpy as np


class BaseModel(object):
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.params = **kwargs
        self.sess = None
        self.loss = None
        self.train_data = None
        self.test_data = None

    def constrct_nn(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def dump(self):
        pass

    def store(self):
        pass
