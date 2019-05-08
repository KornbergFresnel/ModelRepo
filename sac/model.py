import numpy as np
import tensorflow as tf

from lib.common import Buffer, BaseModel, Transition
from lib.tools import flatten


class Episode(Buffer):
    def __init__(self, len_episode):
        super(Episode, self).__init__(len_episode)

    def clear(self):
        self._data = []
        self._flag = 0

    def sample(self, batch_size):
        return Transition(*zip(*self._data))

    def return_cum_reward(self, gamma, value):
        reward = [None for _ in range(len(self._data))]
        reward[-1] = 0 if self._data[-1].done else gamma * value

        for i in range(len(self._data) - 2, 0, -1):
            reward[i] = self._data[i].reward + self.gamma * self._data[i].reward

        return reward


class AC(BaseModel):
    def __init__(self, name, sess, state_space, action_space, len_episode, gamma=0.99, actor_lr=1e-4, critic_lr=1e-3):
        super().__init__(name, state_space, action_space)

        self.sess = sess
        self.len_episode = len_episode
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.act_dim = flatten(self.act_space)

        self.state_ph = tf.placeholder(tf.float32, (None,) + self.state_space, name='state-ph')
        self.act_ph = tf.placeholder(tf.int32, (None,), name='act-ph')
        self.cum_r_ph = tf.placeholder(tf.float32, (None,), name='cum_r-ph')

        with tf.variable_scope("emb"):
            emb_layer = self._emb(self.state_ph)

        with tf.variable_scope("policy"):
            self.actor_scope = tf.get_variable_scope().name
            self.policy_logits = self._construct(emb=emb_layer, out_dim=self.act_dim)
            self.policy = tf.nn.softmax(self.policy_logits, axis=1)

        with tf.variable_scope("value"):
            self.critic_scope = tf.get_variable_scope().name
            self.value = self._construct(emb=emb_layer, out_dim=1)

        with tf.name_scope('optimization'):
            act_one_hot = tf.one_hot(self.act_ph, self.act_dim)
            log_policy = tf.reduce_sum(tf.log(self.policy) * act_one_hot, axis=1)

            self.a_loss = -tf.reduce_sum(log_policy * (tf.stop_gradient(self.value) - self.cum_r))
            self.c_loss = tf.reduce_mean(tf.square(self.value - self.cum_r))

            self.a_train_op = tf.train.AdamOptimizer(self.actor_lr).minimize(self.a_loss)

            # stop update for observation embedding
            c_optimizer = tf.train.AdamOptimizer(self.critic_lr)
            c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.critic_scope)
            grad_vars = optimizer.compute_gradients(self.c_loss, c_vars)

            self.c_train_op = c_optimizer.apply_gradients(grad_vars)

        self.episode = Episode(self.len_episode)

    def _emb(self, input_ph):
        h = tf.layers.dense(input_ph, units=100, activation=tf.nn.relu)
        emb = tf.layers.dense(h, units=64, activation=tf.nn.relu)
        return emb

    def _construct(self, **kwargs):
        h = tf.layers.dense(kwargs['emb'], units=100, activation=tf.nn.relu)
        out = tf.layers.dense(h, units=kwargs['out_dim'])
        return out

    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def act(self, obs, noise=None):
        policy = self.sess.run(self.policy, feed_dict={self.state_ph: [obs]})
        act = np.random.choice(self.act_dim, p=policy[0])
        return act

    def store_transition(self, *args):
        self.episode.push(*args)

    def train(self, **kwargs):
        data = self.episode.sample(None)
        value = self.sess.run(self.value, feed_dict={self.state_ph: data.state})

        cum_reward = self.episode.return_cum_reward(self.gamma, value[-1])

        # train critic
        c_loss, _ = self.sess.run([self.c_loss, self.c_train_op], feed_dict={
            self.state_ph: data.state,
            self.cum_r_ph: cum_reward
        })

        # train actor
        a_loss, _ = self.sess.run([self.a_loss, self.a_train_op], feed_dict={
            self.state_ph: data.state
            self.cum_r_ph: cum_reward
        })

        return {"a_loss": a_loss, "c_loss": c_loss}


class SAC(BaseModel):
    def __init__(self, name, sess, state_space, action_space, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.01, memory_size=10**6, batch_size=64):
        super().__init__(name, state_space, action_space)

        self.sess = sess
        self.gamma = gamma
        self.tau = tau

        self.replay_buffer = Buffer(memory_size)

        raise NotImplementedError

    def _construct(self, **kwargs):
        pass

    def act(self, state, noise=None):
        pass

    def store_transition(self, *args):
        self.replay_buffer(*args)

    def train(self, **kwargs):
        pass

