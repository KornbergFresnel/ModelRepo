import numpy as np
import tensorflow as tf

from lib.common import Buffer, BaseModel
from lib.tools import flatten


class DQN(BaseModel):
    def __init__(self, sess, name, state_space, act_space, lr=1e-3, gamma=0.99, use_double=True, use_dueling=True,
                 tau=0.01, batch_size=64, policy_type='e_greedy', memory_size=10**6):
        super(DQN, self).__init__(name, state_space, act_space)

        self.sess = sess

        self.lr = lr
        self.gamma = gamma
        self.use_double = use_double
        self.use_dueling = use_dueling
        self.tau = tau
        self.act_dim = flatten(act_space)
        self.batch_size = batch_size
        self.policy_type = policy_type

        self.replay_buffer = Buffer(memory_size)

        self.state_ph = tf.placeholder(tf.float32, (None,) + self.state_space, name='state-ph')
        self.target_q_ph = tf.placeholder(tf.float32, (None,), name='target-q-ph')
        self.act_ph = tf.placeholder(tf.int32, (None,), name='act-ph')

        with tf.variable_scope('eval-net'):
            self.eval_scope = tf.get_variable_scope().name
            self.eval_q_tf = self._construct(input_ph=self.state_ph, out_dim=self.act_dim)

            one_hot = tf.one_hot(self.act_ph, self.act_dim)
            self.selected_q_tf = tf.reduce_sum(self.eval_q_tf * one_hot, axis=1, name='selected-q-tf')

        with tf.variable_scope('target-net'):
            self.target_scope = tf.get_variable_scope().name
            self.target_q_tf = self._construct(inupt_ph=self.state_space, out_dim=self.act_dim)

        with tf.name_scope('update'):
            e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.eval_scope)
            t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.target_scope)

            self.async = [tf.assign(t_var, e_var) for t_var, e_var in zip(e_vars, t_vars)]
            self.soft_async = [tf.assign(t_var, self.tau * e_var + (1. - self.tau) * t_var) for t_var, e_var
                               in zip(e_vars, t_vars)]

        with tf.name_scope('optimization'):
            self.loss = 0.5 * tf.reduce_mean(tf.square(self.selected_q_tf - self.target_q_ph))
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.name_scope('cal_target'):
            if self.use_double:
                act = tf.argmax(self.eval_q_tf, axis=1)
                one_hot = tf.one_hot(act, self.act_dim)
                self.reduced_target_tf = tf.reduce_sum(self.target_q_tf * one_hot, axis=1)
            else:
                self.reduced_target_tf = tf.reduce_max(self.target_q_tf, axis=1)

        with tf.name_scope('policy'):
            self.exploration_ph = tf.placeholder(tf.float32, None, name='exploration-ph')

            if self.policy_type == 'e_greedy':
                self.policy = tf.random.uniform(None) < self.exploration_ph
            elif self.policy_type == 'boltzman':
                self.policy = tf.nn.softmax(self.eval_q_tf / self.exploration_ph)
            else:
                raise NotImplementedError

    def _construct(self, **kwargs):
        x = kwargs['input_ph']

        h = tf.layers.dense(x, units=100, activation=tf.nn.relu)
        h = tf.layers.dense(h, units=100, activation=tf.nn.relu)

        out = tf.layers.dense(h, units=kwargs['out_dim'])

        return out

    def act(self, obs, factor=0.):
        policy, value = self.sess.run([self.policy, self.eval_q_tf], feed_dict={
            self.state_ph: [obs],
            self.exploration_ph: factor
        })

        if self.policy_type == 'e_greedy':
            act = np.random.choice(self.act_dim) if policy else np.argmax(value, axis=1)
        elif self.policy_type == 'boltzman':
            act = np.random.choice(self.act_dim, p=policy)
        else:
            raise NotImplementedError

        return act

    def cal_target(self, next_state, reward, done):
        res = self.sess.run(self.reduced_target_tf, feed_dict={self.state_ph: next_state})

        if self.use_dueling:
            raise NotImplementedError

        res = reward + (1. - done) * res * self.gamma

        return res

    def store_transition(self, *args):
        self.replay_buffer.push(*args)

    def train(self, **kwargs):
        batch = self.replay_buffer.sample(self.batch_size)

        target_q = self.cal_target(batch.next_state, batch.reward, batch.done)

        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
            self.state_ph: batch.state,
            self.target_q_ph: target_q
        })

        self.sess.run(self.soft_async)

        return loss
