import time
import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.distributions as tcd
import tensorflow.contrib as tc
from config import GeneralConfig
from tools import Record


class BaseModel(object):
    def __init__(self, name):
        self.name = name
        self._saver = None
        self._sess = None

    def train(self, **kwargs):
        raise NotImplementedError

    def _construct(self, **kwargs):
        raise NotImplementedError


class Actor(BaseModel):
    def __init__(self, env, sess, name, agent_id, config):
        super().__init__(name)

        self._lr = config.actor_lr
        self.test_every = config.test_every
        self._tau = config.update_decay
        self.T = config.temperature

        self.sess = sess
        self.env = env
        self.agent_id = agent_id

        self._layers_conf = config.layers
        self._action_space = env.action_space[agent_id]
        self._observation_space = env.observation_space[agent_id]

        with tf.variable_scope("actor"):
            self.obs_input = tf.placeholder(tf.float32, shape=(None,) + self._observation_space.shape, name="Obs")

            with tf.variable_scope("eval"):
                self._e_scope = tf.get_variable_scope().name
                self.e_out = self._construct(self.action_space.n)
                self._act_prob = tf.nn.softmax(self.e_out / self.T)

            with tf.variable_scope("target"):
                self._t_scope = tf.get_variable_scope().name
                self.t_out = self._construct(self.action_space.n)

            with tf.variable_scope("Optimization"):
                self.q_gradient = tf.placeholder(tf.float32, shape=(None, self.action_space.n), name="Q-gradient")
                self.policy_gradients = tf.gradients(self.e_out, self.e_variables, self.q_gradient)
                self.train_op = tf.train.AdamOptimizer(self._lr).apply_gradients(
                    zip(self.policy_gradients, self.e_variables)
                )

            with tf.variable_scope("Update"):  # smooth average update process
                self._update_op = [tf.assign(t_var, e_var) for t_var, e_var in zip(self.t_variables, self.e_variables)]
                self._soft_update_op = [tf.assign(t_var, self._tau * e_var + (1. - self._tau) * t_var) for t_var, e_var in zip(self.t_variables, self.e_variables)]

    @property
    def t_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._t_scope)

    @property
    def e_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._e_scope)

    def update(self):
        self.sess.run(self._update_op)

    def soft_udpate(self):
        self.sess.run(self._soft_update_op)

    def _construct(self, out_dim, norm=True):
        l1 = tf.layers.dense(self.obs_input, units=self._layers_conf[0], activation=tf.nn.relu, name="l1")
        if norm: l1 = tc.layers.layer_norm(l1)

        l2 = tf.layers.dense(l1, units=self._layers_conf[1], activation=tf.nn.relu, name="l2")
        if norm: l2 = tc.layers.layer_norm(l2)

        out = tf.layers.dense(l2, units=out_dim)

        return out

    def act(self, obs_set):
        policy = self.sess.run(self._act_prob, feed_dict={self.obs_input: [obs_set]})
        act = np.random.choice(self._action_space.n, p=policy[0])

        return act

    def target_act(self, obs_set):
        """Return an action id -> integer"""
        policy = self.sess.run(self.t_out, feed_dict={self.obs_input: obs_set})
        return policy

    def update(self):
        self.sess.run(self.update_op)

    def train(self, obs, action_gradients):
        self.sess.run(self.train_op, feed_dict={
            self.obs_input: obs,
            self.q_gradient: action_gradients
        })


class Critic(BaseModel):
    def __init__(self, env, sess, name, agent_id, config):
        super().__init__(name)

        self.sess = sess
        self.env = env
        self.agent_id = agent_id

        # flatten observation shape
        self.mul_obs_dim = (sum([len(env.observation_callback(env.agents[i], env.world)) for i in range(env.n)]),)
        # flatten action shape
        self.mul_act_dim = (sum([env.action_space[i].n for i in range(env.n)]),)

        self._lr = config.critic_lr
        self.L2 = config.L2
        self.gamma = config.gamma
        self.layers_conf = config.layers
        self.update_every = config.update_every
        self.test_every = config.test_every
        self._tau = config.update_decay

        self.active_func = tf.nn.relu

        with tf.variable_scope("critic"):
            self.mul_obs_input = tf.placeholder(tf.float32, shape=(None,) + self.mul_obs_dim, name="obs-input")
            self.mul_act_input = tf.placeholder(tf.float32, shape=(None,) + self.mul_act_dim, name="mul-act-input")
            self.input = tf.concat([self.mul_obs_input, self.mul_act_input], axis=1, name="concat-input")

            with tf.variable_scope("eval"):
                self._e_scope = tf.get_variable_scope().name
                self.e_q = self._construct()

            with tf.variable_scope("target"):
                self._t_scope = tf.get_variable_scope().name
                self.t_q = self._construct()

            with tf.name_scope("Update"):  # smooth average update process
                self._update_op = [tf.assign(t_var, e_var) for t_var, e_var in zip(self.t_variables, self.e_variables)]
                self._soft_update_op = [tf.assign(t_var, self._tau * e_var + (1. - self._tau) * t_var) for t_var, e_var in zip(self.t_variables, self.e_variables)]

            with tf.variable_scope("Optimization"):
                weight_decay = tf.add_n([self.L2 * tf.nn.l2_loss(var) for var in self.e_variables])
                self.t_q_input = tf.placeholder(tf.float32, shape=(None, 1), name="target-input")
                self.loss = 0.5 * tf.reduce_mean(tf.square(self.t_q_input - self.e_q)) + weight_decay
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                self.action_gradients = tf.gradients(-tf.reduce_mean(self.e_q), self.mul_act_input)

    @property
    def t_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._t_scope)

    @property
    def e_variables(self):
        return tf.get_collection(tf.GraphKeys.METRIC_VARIABLES, scope=self._e_scope)

    def update(self):
        self.sess.run(self._update_op)

    def soft_udpate(self):
        self.sess.run(self._soft_update_op)

    def _construct(self, name_scope, norm=True):
        l1 = tf.layers.dense(self.input, units=self.layers_conf[0], activation=tf.nn.relu, name="l1")
        if norm: l1 = tc.layers.layer_norm(l1)

        l2 = tf.layers.dense(l1, units=self.layers_conf[1], activation=tf.nn.relu, name="l2")
        if norm: l2 = tc.layers.layer_norm(l2)

        out = tf.layers.dense(l2, units=1, name="Q")

        return out

    def calculate_target_q(self, obs_next, action_next):
        q_values = self.sess.run(self.t_q, feed_dict={
            self.mul_obs_input: obs_next,
            self.mul_act_input: action_next
        })

        target_q_value = np.max(q_values, axis=1)

        return target_q_value * self.gamma

    def get_action_graidents(self, obs, action):
        action_gradients, e_q = self.sess.run([self.action_gradients, self.e_q], feed_dict={
            self.mul_obs_input: obs,
            self.mul_act_input: action
        })
        return action_gradients[0], np.mean(e_q)

    def train(self, target_q_values, obs, action):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.mul_obs_input: obs,
            self.mul_act_input: action,
            self.t_q_input: target_q_values.reshape((target_q_values.shape[0], 1)),
        })

        return loss


class MultiAgent(object):
    def __init__(self, env, name):
        # == Initialize ==
        self.name = name
        self.sess = tf.Session()
        self.env = env

        self.actor = []  # hold all Actors
        self.critic = []  # hold all Critics
        self.actions_dims = []  # record the action split for gradient apply

        # == Construct Network for Each Agent ==
        with tf.variable_scope(self.name):
            for agent_id in range(self.env.n):
                with tf.variable_scope(name + "_{}".format(agent_id)):
                    self.actor.append(Actor(env, self.sess, name, agent_id, config))
                    self.critic.append(Critic(env, self.sess, name, agent_id, config))

                    self.actions_dims.append(self.env.action_space[agent_id].n)

        self.sess.run(tf.global_variables_initializer())

        for i in range(self.env.n):
            self.actor[i].update()
            self.critic[i].update()

    def store_trans(self, **kwargs):
        raise NotImplementedError

    def act(self, obs_set, noise=0.0):
        """Accept a observation list, return action list of all agents."""
        actions = []
        for i, obs in enumerate(obs_set):
            n = self.actions_dims[i]
            actions.append(self.actor[i].act(obs) + np.random.randn(n) * noise)
        return actions

    def restructure(self, data):
        """Restructure in-data for graidients apply

        Arguments
        ---------
        data: list, action gradients from all critics, each element represents action-gradients form agent_i,
            and dimension of each element is: [n_batch, sum_action_dim]

        Returns
        -------
        new_data: list, split original action-gradients with action length of each agent, then restructure them
            for each agents. Dimension of each element is: [n_batch, action_dim_of_agent]
        """
        new_data = [0.0 for _ in range(len(data))]

        begin = 0
        for j, len_act in enumerate(self.actions_dims):
            for i in range(self.env.n):
                content = data[i][:, begin:begin + len_act]
                new_data[j] += content
            begin += len_act

        return new_data

    def async_update(self):
        for j in range(self.env.n):
            self.actor[j].soft_update()
            self.critic[j].soft_update()

    def save(self, dir_path, epoch):
        """Save model

        Arguments
        ---------
        dir_path: str, the grandparent directory which stores all models
        epoch: int, number of current round
        """
        dir_name = os.path.join(dir_path, self.name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, dir_name + "/{}_model_{}.ckpt".format(self.name, epoch))
        print("[*] Model saved in file: {}".format(save_path))

    def load(self, dir_path, epoch=0):
        """Load model from local storage, if no such model file, it will print warning to you

        Arguments
        ---------
        dir_path: str, the grandparent directory which stores all models
        epoch: int, the index which used for indicating a certain model file
        """
        try:
            dir_name = os.path.join(dir_path, self.name)
            model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            saver = tf.train.Saver(model_vars)
            file_path = os.path.join(dir_name, "{}_model_{}.ckpt".format(self.name, epoch))
            saver.restore(self.sess, file_path)
        except Exception as e:
            print("[!] Load model falied, please check {} exists".format(file_path))
            exit(1)

    def train_step(self, batch):
        loss = [0.] * self.env.n
        obs_clus = np.concatenate(batch.obs, axis=1)
        obs_next_clus = np.concatenate(batch.obs_next, axis=1)
        act_clus = np.concatenate(batch.act, axis=1)

        batch_act_next = []

        for j in range(self.env.n):
            batch_act_next.append(self.actor[j].target_act(batch.obs_next[j]))

        batch_act_next = np.concatenate(batch_act_next, axis=1)

        for j in range(self.env.n):
            batch_q = self.critic[j].calculate_target_q(obs_next_clus, batch_act_next)
            batch_q = batch.reward[:, j] + (1. - batch.terminate[:, j]) * batch_q

            critic_loss = self.critic[j].train(batch_q, obs_clus, act_clus)
            action_gradients[j], mean_q = self.critic[j].get_action_graidents(obs_next_clus, batch_act_next)

            loss[j] += critic_loss

        action_grads = self.restructure(np.array(action_gradients))

        for j in range(self.env.n):
            self.actor[j].train(batch.obs_next[j], action_grads[j])

        return loss

    def train(self):
        n_batch = 1
        loss = []

        for _ in range(n_batch):
            batch = self.replay_buffer.sample(self.batch_size)
            temp = self.train_step(batch)
            loss.append(temp)

        return np.mean(loss)
