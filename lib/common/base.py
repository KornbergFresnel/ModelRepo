import os
import tensorflow as tf


class BaseModel(object):
    def __init__(self, name, state_space, action_space):
        self.name = name
        self.sess = None
        self.state_space = state_space
        self.action_space = action_space

    def _construct(self, **kwargs):
        raise NotImplementedError

    def train(self, **kwargs):
        raise NotImplementedError

    def save(self, dir_path, epoch):
        """ Save model
        :param dir_path: str, the grandparent directory path for model saving
        :param epoch: int, global step
        """

        dir_name = os.path.join(dir_path, self.name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, dir_name + "/{}".format(self.name), global_step=epoch)
        print("[*] Model saved in file: {}".format(save_path))

    def load(self, dir_path, epoch=0):
        """ Load model from local storage, if no such file, it will throw an exception.

        :param dir_path: str, the grandparent directory path for model saving
        :param epoch: int, global step
        """

        file_path = None

        try:
            dir_name = os.path.join(dir_path, self.name)
            model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            saver = tf.train.Saver(model_vars)
            file_path = os.path.join(dir_name, "{}-{}".format(self.name, epoch))
            saver.restore(self.sess, file_path)
        except Exception as e:
            print("[!] Load model failed, please check {} exists".format(file_path))
            exit(1)
