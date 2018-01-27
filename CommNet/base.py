import tensorflow as tf
import numpy as np


class BaseModel:
    def __init__(self, num_leaver=5, num_agents=500, vector_len=128, num_units=10, learning_rate=0.003,
                 batch_size=64, episodes=500):
        # ===== parameters =====
        self.num_leaver = num_leaver
        self.num_agents = num_agents
        self.alpha = learning_rate
        self.vector_len = vector_len
        self.num_units = num_units
        self.batch_size = batch_size
        self.n_actions = num_leaver
        self.episodes = episodes

        # ===== pre-define data: look-up table, id =====
        self.ids = None
        self.mask_data = np.ones(shape=(self.num_leaver, self.num_leaver), dtype=np.float32)
        self.mask_data[np.arange(self.num_leaver), np.arange(self.num_leaver)] = 0.0

        # ===== network define =====
        self.input = tf.placeholder(tf.int32, shape=(None, self.num_leaver))
        self.c_meta = tf.placeholder(tf.float32, shape=(None, self.num_leaver, self.vector_len))

    def _create_cell(self, name, c, h, h_meta):
        with tf.variable_scope(name):
            self.H = tf.get_variable("w_h", shape=(self.vector_len, self.vector_len),
                                     initializer=tf.random_normal_initializer())
            self.C = tf.get_variable("w_c", shape=(self.vector_len, self.vector_len),
                                     initializer=tf.random_normal_initializer())
            self.H_META = tf.get_variable("w_h_meta", shape=(self.vector_len, self.vector_len),
                                          initializer=tf.random_normal_initializer())

        dense_h = tf.einsum("ijk,kl->ijl", h, self.H)
        dense_c = tf.einsum("ijk,kl->ijl", c, self.C)
        dense_h_meta = tf.einsum("ijk,kl->ijl", h_meta, self.H_META)

        dense = dense_h + dense_c + dense_h_meta

        return tf.nn.relu(dense)

    def _mean(self, h):
        amount = self.num_leaver - 1

        self.mask = tf.placeholder(tf.float32, shape=(self.num_leaver, self.num_leaver))

        c = tf.einsum("ij,kjl->kil", self.mask, h) / amount

        return c

    def _create_network(self):
        pass

    def _sample_action(self):
        pass

    def get_reward(self, ids_one_hot):
        pass

    def _get_loss(self):
        pass

    def train(self, ids, base_line=None, base_reward=None, **kwargs):
        pass
