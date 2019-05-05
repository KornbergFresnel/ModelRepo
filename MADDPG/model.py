import time
import os
import random
import tensorflow as tf
import numpy as np
import tensorflow.contrib.distributions as tcd
import tensorflow.contrib as tc

from config import GeneralConfig
from tools import Record
from lib.common.memory import Buffer, Transition


class BunchBuffer(Buffer):
    def __init__(self, n_agent, capacity):
        super().__init__(capacity)

        self.n_agent = n_agent
        self._data = [[] for _ in range(self.n_agent)]
        self._size = 0

    def push(self, *args):
        for i, state, action, next_state, reward, done in enumerate(zip(*args)):
            if len(self._data[i]) < self._capacity:
                self._data[i].append(None)

            self._data[i][self._flag] = Transition(state, action, next_state, reward, done)
            self._flag = (self._flag + 1) % self._capacity
            self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size):
        if self._size < batch_size:
            return None

        samples = [None for _ in range(self.n_agent)]

        random.seed(a=self._flag)
        for i in range(self.n_agent):
            tmp = random.sample(self._data[i], batch_size)
            samples[i] = Transition(*zip(*tmp))

        return samples


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
    def __init__(self, sess, state_space, act_space, lr=1e-4, tau=0.01, name=None, agent_id=None):
        super().__init__(name)

        self._lr = lr
        self._tau = tau

        self.sess = sess
        self.agent_id = agent_id

        self._action_space = act_space
        self._observation_space = state_space

        with tf.variable_scope("actor"):
            self.obs_input = tf.placeholder(tf.float32, shape=(None,) + self._observation_space.shape, name="Obs")

            with tf.variable_scope("eval"):
                self._eval_scope = tf.get_variable_scope().name
                self.eval_net = self._construct(self._action_space.n)
                self._act_prob = tf.nn.softmax(self.eval_net)

                self._act_input = tf.placeholder(tf.int32, shape=(None,), name="act-input")
                self._act_tf = tf.one_hot(self._act_input, self._action_space.n) * self._act_prob

            # !!!Deprecated
            with tf.variable_scope("target"):
                self._target_scope = tf.get_variable_scope().name
                self.t_out = self._construct(self._action_space.n)

            with tf.variable_scope("Update"):  # smooth average update process
                self._update_op = [tf.assign(t_var, e_var) for t_var, e_var in zip(self.t_variables, self.e_variables)]
                self._soft_update_op = [tf.assign(t_var, self._tau * e_var + (1. - self._tau) * t_var) for t_var, e_var in zip(self.t_variables, self.e_variables)]

    @property
    def t_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._target_scope)

    @property
    def e_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._eval_scope)

    @property
    def act_tensor(self):
        return self._act_tf

    @property
    def obs_tensor(self):
        return self.obs_input

    def _construct(self, out_dim, norm=True):
        l1 = tf.layers.dense(self.obs_input, units=100, activation=tf.nn.relu, name="l1")
        if norm: l1 = tc.layers.layer_norm(l1)

        l2 = tf.layers.dense(l1, units=100, activation=tf.nn.relu, name="l2")
        if norm: l2 = tc.layers.layer_norm(l2)

        out = tf.layers.dense(l2, units=out_dim)

        return out

    def set_optimization(self, q_func):
        with tf.variable_scope("optimization"):
            self._loss = -tf.reduce_mean(q_func.value)
            optimizer = tf.train.AdamOptimizer(self._lr)
            grad_vars = optimizer.compute_gradients(self._loss, self.e_variables)
            self._train_op = optimizer.apply_gradients(grad_vars)

    def update(self):
        self.sess.run(self._update_op)

    def soft_udpate(self):
        self.sess.run(self._soft_update_op)

    def act(self, obs):
        policy = self.sess.run(self._act_prob, feed_dict={self.obs_input: [obs]})
        act = np.random.choice(self._action_space.n, p=policy[0])

        return act

    def target_act(self, obs):
        """Return an action id -> integer"""
        policy = self.sess.run(self.t_out, feed_dict={self.obs_input: obs_set})
        return policy

    def train(self, obs):
        # TODO(ming): need to refine
        self.sess.run(self._train_op, feed_dict={
            self.obs_input: obs
        })


class Critic(BaseModel):
    def __init__(self, sess, multi_obs_phs, multi_act_phs, lr=1e-3, gamma=0.98, tau=0.01, name=None, agent_id=None):
        super().__init__(name)

        self.sess = sess
        self.agent_id = agent_id

        # flatten observation shape
        # self.mul_obs_dim = (sum([len(env.observation_callback(env.agents[i], env.world)) for i in range(env.n)]),)
        self.mul_obs_dim = None
        # flatten action shape
        # self.mul_act_dim = (sum([env.action_space[i].n for i in range(env.n)]),)
        self.mul_act_dim = None

        self._lr = lr
        self._tau = 0.01
        self.L2 = 0.
        self.gamma = gamma

        with tf.variable_scope("critic"):
            # TODO(ming): transform the list of tensor into TensorArray, both for obs and act
            self.mul_obs_input = tf.placeholder(tf.float32, shape=(None,) + self.mul_obs_dim, name="obs-input")
            self.mul_act_input = tf.concat(multi_act_phs, axis=1, name="act-input")
            self.input = tf.concat([self.mul_obs_input, self.mul_act_input], axis=1, name="concat-input")
            self.target_input = None

            with tf.variable_scope("eval"):
                self._e_scope = tf.get_variable_scope().name
                self.e_q = self._construct(self.input)

            with tf.variable_scope("target"):
                self._t_scope = tf.get_variable_scope().name
                self.t_q = self._construct(self.target_input)

            with tf.name_scope("Update"):  # smooth average update process
                self._update_op = [tf.assign(t_var, e_var) for t_var, e_var in zip(self.t_variables, self.e_variables)]
                self._soft_update_op = [tf.assign(t_var, self._tau * e_var + (1. - self._tau) * t_var) for t_var, e_var in zip(self.t_variables, self.e_variables)]

            with tf.variable_scope("Optimization"):
                weight_decay = tf.add_n([self.L2 * tf.nn.l2_loss(var) for var in self.e_variables])
                self.t_q_input = tf.placeholder(tf.float32, shape=(None, 1), name="target-input")
                self.loss = 0.5 * tf.reduce_mean(tf.square(self.t_q_input - self.e_q)) + weight_decay
                self.train_op = tf.train.AdamOptimizer(self._lr).minimize(self.loss)

    @property
    def t_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._t_scope)

    @property
    def e_variables(self):
        return tf.get_collection(tf.GraphKeys.METRIC_VARIABLES, scope=self._e_scope)

    @property
    def value(self):
        return self.e_q

    @property
    def obs_cluster_ph(self):
        return self.mul_obs_input

    @property
    def act_cluster_ph(self):
        return self.mul_act_input

    def _construct(self, input_ph, norm=True):
        l1 = tf.layers.dense(input_ph, units=100, activation=tf.nn.relu, name="l1")
        if norm: l1 = tc.layers.layer_norm(l1)

        l2 = tf.layers.dense(l1, units=100, activation=tf.nn.relu, name="l2")
        if norm: l2 = tc.layers.layer_norm(l2)

        out = tf.layers.dense(l2, units=1, name="Q")

        return out

    def update(self):
        self.sess.run(self._update_op)

    def soft_udpate(self):
        self.sess.run(self._soft_update_op)

    def calculate_target_q(self, obs_next, action_next):
        q_values = self.sess.run(self.t_q, feed_dict={
            self.mul_obs_input: obs_next,
            self.mul_act_input: action_next
        })

        target_q_value = np.max(q_values, axis=1)

        return target_q_value * self.gamma

    def train(self, target_q_values, obs, action):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.mul_obs_input: obs,
            self.mul_act_input: action,
            self.t_q_input: target_q_values.reshape((target_q_values.shape[0], 1)),
        })

        return loss


class MultiAgent(object):
    def __init__(self, env, name, n_agent, batch_size=64, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.01, memory_size=10**4):
        # == Initialize ==
        self.name = name
        self.sess = tf.Session()
        self.env = env
        self.n_agent = n_agent

        self.actors = []  # hold all Actors
        self.critics = []  # hold all Critics
        self.actions_dims = []  # record the action split for gradient apply

        self.replay_buffer = Buffer(memory_size)
        self.batch_size = batch_size

        # == Construct Network for Each Agent ==
        with tf.variable_scope(self.name):
            for agent_id in range(self.env.n):
                with tf.name_scope("policy_{}_{}".format(name, agent_id)):
                    self.actors.append(Actor(env, self.sess, name, agent_id))

            # collect action outputs of all actors
            ori_act_and_obs_phs = [(actor.act_tensor, actor.obs_tensor) for actor in self.actors]

            for agent_id in range(self.env.n):
                with tf.name_scope("critic_{}_{}".format(name, agent_id)):
                    act_phs = self._mask_other_act_phs(ori_act_and_obs_phs[0], agent_id)
                    self.critics.append(Critic(env, self.sess, name, agent_id, act_phs))
                    self.actions_dims.append(self.env.action_space[agent_id].n)

            # set optimization for actors
            for actor, critic in zip(self.actors, self.critics):
                with tf.name_scope("optimizer_{}_{}".format(name, agent_id)):
                    actor.set_optimization(critic)

        self.sess.run(tf.global_variables_initializer())

        # hard sync
        for i in range(self.env.n):
            self.actors[i].update()
            self.critics[i].update()

    def _mask_other_act_phs(self, act_phs, agent_id):
        res = []
        for i, ph in enumerate(act_phs):
            if agent_id == i:
                res.append(ph)
            else:
                res.append(tf.stop_gradient(ph))

        return res

    def store_trans(self, **kwargs):
        raise NotImplementedError

    def act(self, obs_set, noise=0.0):
        """Accept a observation list, return action list of all agents."""
        actions = []
        for i, (obs, agent) in enumerate(zip(obs_set, self.actors)):
            n = self.actions_dims[i]
            actions.append(agent.act(obs) + np.random.randn(n) * noise)
        return actions

    def async_update(self):
        for j in range(self.env.n):
            self.actors[j].soft_update()
            self.critics[j].soft_update()

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

    def train_step(self, batch_list):
        loss = [0.] * self.n_agent

        state_clus = [None for _ in range(self.n_agent)]
        next_state_clus = [None for _ in range(self.n_agent)]
        act_clus = [None for _ in range(self.n_agent)]

        for i, batch in enumerate(batch_list):
            state_clus[i] = np.concatenate(batch.state, axis=1)
            next_state_clus[i] = np.concatenate(batch.next_state, axis=1)
            act_clus[i] = np.concatenate(batch.action, axis=1)

        batch_act_next = []

        for j in range(self.env.n):
            batch_act_next.append(self.actors[j].target_act(batch.obs_next[j]))

        batch_act_next = np.concatenate(batch_act_next, axis=1)

        for j in range(self.env.n):
            batch_q = self.critics[j].calculate_target_q(next_state_clus, batch_act_next)
            batch_q = batch.reward[:, j] + (1. - batch.terminate[:, j]) * batch_q

            critic_loss = self.critics[j].train(batch_q, obs_clus, act_clus)
            loss[j] += critic_loss

        for j in range(self.env.n):
            self.actors[j].train(batch.obs_next[j])

        return loss

    def train(self):
        n_batch = 1
        loss = []

        for _ in range(n_batch):
            batch = self.replay_buffer.sample(self.batch_size)
            temp = self.train_step(batch)
            loss.append(temp)

        return np.mean(loss)
