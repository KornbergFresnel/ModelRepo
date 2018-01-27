import numpy as np
import tensorflow as tf
from base import BaseModel


class CommNet(BaseModel):
    def __init__(self, num_leaver=5, num_agents=500, vector_len=128, num_units=10, learning_rate=0.0005, batch_size=64,
                 episodes=500):

        super().__init__(num_leaver, num_agents, vector_len, num_units, learning_rate, batch_size, episodes)

        self.base_line = tf.placeholder(tf.float32, shape=(None, 1))
        self.base_reward = tf.placeholder(tf.float32, shape=(None, 1))
        self.bias = 1e-4

        # ==== create network =====
        with tf.variable_scope("CommNet"):
            self.eval_name = tf.get_variable_scope().name
            self.look_up = tf.get_variable("look_up_table", shape=(self.num_agents, self.vector_len),
                                           initializer=tf.random_normal_initializer)
            self.dense_weight = tf.get_variable("dense_w", shape=(self.vector_len, self.n_actions),
                                                initializer=tf.random_uniform_initializer)
            self.policy = self._create_network()

            self.reward = self._get_reward()
            self.loss = self._get_loss()
            self.train_op = tf.train.RMSPropOptimizer(self.alpha).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _create_network(self):
        # look-up table

        input_one_hot = tf.one_hot(self.input, self.num_agents)
        # CommNet
        h0 = tf.einsum("ijk,kl->ijl", input_one_hot, self.look_up)
        h1 = self._create_cell("step_first", self.c_meta, h0, h0)
        c1 = self._mean(h1)
        h2 = self._create_cell("step_second", c1, h1, h0)
        out = tf.einsum("ijk,kl->ijl", h2, self.dense_weight)

        # soft-max
        soft = tf.nn.softmax(out)

        return soft

    def _sample_action(self):
        reshape_policy = tf.reshape(self.policy, shape=(-1, self.n_actions))
        # sample actions
        self.actions = tf.multinomial(tf.log(reshape_policy + self.bias), num_samples=1)
        one_hot = tf.one_hot(self.actions, depth=self.n_actions)
        self.one_hot = tf.reshape(one_hot, shape=(-1, self.num_leaver, self.n_actions))

    def _get_reward(self):
        self._sample_action()
        distinct_num = tf.reduce_sum(tf.cast(tf.reduce_sum(self.one_hot, axis=1) > 0, tf.float32), axis=1,
                                     keep_dims=True)
        return distinct_num / self.num_leaver

    def _get_loss(self):
        # advantage: n, 1, 1
        meta = tf.reshape(self.base_reward - self.base_line, shape=(-1, 1, 1))
        labels = tf.reshape(tf.cast(self.one_hot, dtype=tf.float32) * tf.tile(meta, [1, 5, 5]),
                            shape=(-1, self.n_actions))
        prob = tf.reshape((self.policy + self.bias), shape=(-1, self.n_actions))
        loss = tf.reduce_mean(tf.reduce_sum(-1.0 * labels * tf.log(prob), axis=1))

        return loss

    def get_reward(self, ids):
        # produce
        reward, gun, policy = self.sess.run([self.reward, self.dense_weight, self.policy], feed_dict={
            self.input: ids,
            self.mask: self.mask_data,
            self.c_meta: np.zeros((self.batch_size, self.num_leaver, self.vector_len))
        })

        return reward

    def train(self, ids, base_line=None, base_reward=None, **kwargs):

        _, loss, reward, policy = self.sess.run([self.train_op, self.loss, self.reward, self.policy], feed_dict={
            self.input: ids,
            self.mask: self.mask_data,
            self.c_meta: np.zeros((self.batch_size, self.num_leaver, self.vector_len)),
            self.base_line: base_line,
            self.base_reward: base_reward
        })

        log = kwargs["log"]
        itr = kwargs["itr"]

        sum_loss = np.sum(loss) / self.batch_size
        sum_base = np.sum(reward) / self.batch_size

        if (itr + 1) % 20 == 0:
            log.info("iteration:{0}\tloss:{1}\treward:{2}".format(itr, sum_loss, sum_base))


class BaseLine(BaseModel):
    def __init__(self, num_leaver=5, num_agents=500, vector_len=128, num_units=10, learning_rate=0.0005, batch_size=64,
                 episodes=500):
        super().__init__(num_leaver, num_agents, vector_len, num_units, learning_rate, batch_size, episodes)

        self.n_actions = 1
        self.eta = 0.003

        self.reward = tf.placeholder(tf.float32, shape=(None, 1))

        # ==== create network =====
        with tf.variable_scope("Arctic"):
            self.dense_weight = tf.get_variable("dense_weight", shape=(self.vector_len, 1),
                                                initializer=tf.random_normal_initializer)
            self.baseline = self._create_network()  # n * 5 * n_actions
            # cross entropy: n * 5 * 1
            self.loss = self._get_loss()
            self.train_op = tf.train.RMSPropOptimizer(self.alpha).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _create_network(self):
        # look-up table
        look_up = tf.get_variable("look_up_table", shape=(self.num_agents, self.vector_len),
                                  initializer=tf.random_normal_initializer)

        input_one_hot = tf.one_hot(self.input, self.num_agents)
        # CommNet-Baseline
        h0 = tf.einsum("ijk,kl->ijl", input_one_hot, look_up)
        h1 = self._create_cell("step_first", self.c_meta, h0, h0)
        c1 = self._mean(h1)
        h2 = self._create_cell("step_second", c1, h1, h0)

        dense = tf.einsum("ijk,kl->ijl", h2, self.dense_weight)

        # out = tf.einsum("ijk,kl->ijl", dense)

        self.t = tf.sigmoid(dense)

        out = tf.reduce_mean(self.t, axis=1)

        return out

    def _get_loss(self):
        loss = tf.reduce_sum(tf.square(self.reward - self.baseline)) * self.eta
        return loss

    def get_reward(self, ids):
        return self.sess.run(self.baseline, feed_dict={
            self.input: ids,
            self.mask: self.mask_data,
            self.c_meta: np.zeros((self.batch_size, self.num_leaver, self.vector_len))
        })

    def train(self, ids, base_line=None, base_reward=None, **kwargs):

        _, loss, base = self.sess.run([self.train_op, self.loss, self.baseline], feed_dict={
            self.input: ids,
            self.mask: self.mask_data,
            self.c_meta: np.zeros((self.batch_size, self.num_leaver, self.vector_len)),
            self.reward: base_reward
        })

        log = kwargs["log"]
        itr = kwargs["itr"]

        sum_loss = np.sum(loss) / self.batch_size
        sum_base = np.sum(base) / self.batch_size

        if (itr + 1) % 20 == 0:
            log.info("iteration:{0}\tloss:{1}\tbase:{2}".format(itr, sum_loss, sum_base))
            # print("iteration:{0}\tloss:{1}\tbase:{2}".format(itr, sum_loss, sum_base))







