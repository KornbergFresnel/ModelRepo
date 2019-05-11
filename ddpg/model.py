import numpy as np
import tensorflow as tf

from lib.common import Buffer, BaseModel, Transition
from lib.tools import flatten, softmax


class DDPG(BaseModel):
    def __init__(self, name, sess, state_space, action_space, gamma=0.98, actor_lr=1e-4, critic_lr=1e-3, memory_size=10**3, batch_size=64, tau=0.01, grad_norm=5.0):
        super(DDPG, self).__init__(name, state_space, action_space)

        self.sess = sess
        self.gamma = gamma
        self.a_lr = actor_lr
        self.c_lr = critic_lr
        self.tau = tau
        self.batch_size = batch_size

        self.act_dim = flatten(action_space)
        self.replay_buffer = Buffer(memory_size)

        self.state_ph = tf.placeholder(tf.float32, (None,) + state_space, name='state-ph')
        self.next_state_ph = tf.placeholder(tf.float32, (None,) + state_space, name='next_state-ph')
        self.reward_ph = tf.placeholder(tf.float32, (None,), name='reward-ph')
        self.done_ph = tf.placeholder(tf.float32, (None,), name='done-ph')

        with tf.variable_scope('policy'):
            p_scope = tf.get_variable_scope().name
            self.logits = self._construct(input_ph=self.state_ph, out_dim=self.act_dim)
            self.policy = tf.nn.softmax(self.logits)

        with tf.variable_scope('target_policy'):
            t_p_scope = tf.get_variable_scope().name
            self.target_logits = self._construct(input_ph=self.next_state_ph, out_dim=self.act_dim)
            self.target_policy = tf.nn.softmax(self.target_logits)

        with tf.variable_scope('value'):
            q_scope = tf.get_variable_scope().name
            self.q = self._construct(input_ph=tf.concat([self.state_ph, self.policy], axis=1), out_dim=1)

        with tf.variable_scope('target_value'):
            t_q_scope = tf.get_variable_scope().name
            self.target_q = self._construct(input_ph=tf.concat([self.next_state_ph, self.target_policy], axis=1), out_dim=1)

            self.next_q = gamma * (1. - self.done_ph) * tf.reshape(self.target_q, (-1,)) + self.reward_ph

        with tf.name_scope('update'):
            e_p_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=p_scope)
            t_p_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=t_p_scope)

            e_q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
            t_q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=t_q_scope)

            self.sync = [tf.assign(t_var, e_var) for e_var, t_var in zip(e_p_vars, t_p_vars)] + [tf.assign(t_var, e_var) for e_var, t_var in zip(e_q_vars, t_q_vars)]

            self.soft_sync = [tf.assign(t_var, tau * e_var + (1. - tau) * t_var) for e_var, t_var in zip(e_p_vars, t_p_vars)] + [tf.assign(t_var, tau * e_var + (1. - tau) * t_var) for e_var, t_var in zip(e_q_vars, t_q_vars)]

        with tf.name_scope('optimization'):
            policy_loss = -tf.reduce_mean(self.q)
            value_loss = tf.reduce_mean(tf.square(tf.stop_gradient(self.next_q) - tf.reshape(self.q, (-1,))))

            optimizer = tf.train.AdamOptimizer(self.a_lr)
            grad_vars = optimizer.compute_gradients(policy_loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=p_scope))
            grad_vars = [(tf.clip_by_value(grad, -1., 1.), _var) for grad, _var in grad_vars]
            self.p_train = optimizer.apply_gradients(grad_vars)

            optimizer = tf.train.AdamOptimizer(self.c_lr)
            grad_vars = optimizer.compute_gradients(value_loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope))
            self.c_train = optimizer.apply_gradients(grad_vars)

            self.a_loss = policy_loss
            self.c_loss = value_loss

    def _construct(self, **kwargs):
        x = kwargs['input_ph']

        h = tf.layers.dense(x, units=100, activation=tf.nn.relu)
        h = tf.layers.dense(h, units=100, activation=tf.nn.relu)

        out = tf.layers.dense(h, units=kwargs['out_dim'])

        return out

    def sync_net(self):
        self.sess.run(self.sync)

    def act(self, state, noise=0.):
        logits = self.sess.run(self.logits, feed_dict={self.state_ph: [state]})
        logits = logits[0] + noise

        policy = softmax(logits)

        return np.random.choice(self.act_dim, p=policy)

    def store_transition(self, *args):
        self.replay_buffer.push(*args)

    def train(self, **kwargs):
        if len(self.replay_buffer) < self.batch_size:
            return None

        n_batch = len(self.replay_buffer) // self.batch_size
        mean_a_loss, mean_c_loss = 0., 0.

        for _ in range(n_batch):
            data = self.replay_buffer.sample(self.batch_size)

            c_loss, _ = self.sess.run([self.c_loss, self.c_train], feed_dict={
                self.state_ph: data.state,
                self.reward_ph: data.reward,
                self.done_ph: data.done,
                self.next_state_ph: data.next_state
            })

            a_loss, _ = self.sess.run([self.a_loss, self.p_train], feed_dict={
                self.state_ph: data.state
            })

            mean_a_loss += a_loss
            mean_c_loss += c_loss

            self.sess.run(self.soft_sync)

        return mean_a_loss / n_batch, mean_c_loss / n_batch
