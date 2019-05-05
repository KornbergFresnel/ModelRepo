import os
import random
import operator
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tc

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

        self._loss = None
        self._train_op = None

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

    def soft_update(self):
        self.sess.run(self._soft_update_op)

    def act(self, obs):
        policy = self.sess.run(self._act_prob, feed_dict={self.obs_input: [obs]})
        act = np.random.choice(self._action_space.n, p=policy[0])

        return act

    def target_act(self, obs):
        """ Return an action id -> integer """

        policy = self.sess.run(self.t_out, feed_dict={self.obs_input: obs})

        return policy

    def train(self, obs):
        loss, _ = self.sess.run([self._loss, self._train_op], feed_dict={
            self.obs_input: obs
        })

        return loss


class Critic(BaseModel):
    def __init__(self, sess, multi_obs_phs, multi_act_phs, lr=1e-3, gamma=0.98, tau=0.01, name=None, agent_id=None):
        super().__init__(name)

        self.sess = sess
        self.agent_id = agent_id

        # flatten observation shape
        self.mul_obs_dim = None
        # flatten action shape
        self.mul_act_dim = None

        self._lr = lr
        self._tau = tau
        self.L2 = 0.
        self.gamma = gamma

        with tf.variable_scope("critic"):
            self.multi_obs_phs = multi_obs_phs
            self.multi_act_phs = multi_act_phs

            obs_input = tf.concat(multi_obs_phs, axis=1, name="obs-clus-input")
            act_input = tf.concat(multi_act_phs, axis=1, name="act-clus-input")

            self.input = tf.concat([obs_input, act_input], axis=1, name="concat-input")
            self.target_input = None

            with tf.variable_scope("eval"):
                self._e_scope = tf.get_variable_scope().name
                self.e_q = self._construct(self.input)

            with tf.variable_scope("target"):
                self._t_scope = tf.get_variable_scope().name
                self.t_q = self._construct(self.target_input)

            with tf.name_scope("Update"):  # smooth average update process
                self._update_op = [tf.assign(t_var, e_var) for t_var, e_var in zip(self.t_variables, self.e_variables)]
                self._soft_update_op = [tf.assign(t_var, self._tau * e_var + (1. - self._tau) * t_var) for t_var, e_var
                                        in zip(self.t_variables, self.e_variables)]

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
        return self.multi_obs_phs

    @property
    def act_cluster_ph(self):
        return self.multi_act_phs

    def _construct(self, input_ph, norm=True):
        l1 = tf.layers.dense(input_ph, units=100, activation=tf.nn.relu, name="l1")
        if norm: l1 = tc.layers.layer_norm(l1)

        l2 = tf.layers.dense(l1, units=100, activation=tf.nn.relu, name="l2")
        if norm: l2 = tc.layers.layer_norm(l2)

        out = tf.layers.dense(l2, units=1, name="Q")

        return out

    def update(self):
        self.sess.run(self._update_op)

    def soft_update(self):
        """ Soft update target net """

        self.sess.run(self._soft_update_op)

    def calculate_target_q(self, next_obs_list, next_act_list):
        """ Return target Q value

        :param next_obs_list: list, a list of N agents' observations
        :param next_act_list: list, a list of N agents' actions
        :return: array-like with shape: (n_batch, 1)
        """

        feed_dict = dict()

        feed_dict.update(zip(self.multi_obs_phs, next_obs_list))
        feed_dict.update(zip(self.multi_act_phs, next_act_list))

        q_values = self.sess.run(self.t_q, feed_dict=feed_dict)

        return q_values

    def train(self, target_q_values, obs_list, act_list):
        """ Train critic network

        :param target_q_values: array-like with shape of (n_batch, 1)
        :param obs_list: list, list of N agents' observations
        :param act_list: list, list of N agents' actions
        :return: critic loss, float
        """

        feed_dict = dict()

        feed_dict.update(zip(self.multi_obs_phs, obs_list))
        feed_dict.update(zip(self.multi_act_phs, act_list))
        feed_dict[self.t_q_input] = target_q_values

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)

        return loss


class MultiAgent(object):
    def __init__(self, sess, env, name, n_agent, batch_size=64, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.01, memory_size=10**4):
        # == Initialize ==
        self.name = name
        self.sess = sess
        self.env = env
        self.n_agent = n_agent
        self.gamma = gamma

        self.actors = []  # hold all Actors
        self.critics = []  # hold all Critics
        self.actions_dims = []  # record the action split for gradient apply

        self.replay_buffer = Buffer(memory_size)
        self.batch_size = batch_size

        # == Construct Network for Each Agent ==
        with tf.variable_scope(self.name):
            for agent_id in range(self.env.n):
                with tf.name_scope("policy_{}_{}".format(name, agent_id)):
                    self.actors.append(Actor(self.sess, state_space=None, act_space=None, lr=actor_lr, tau=tau,
                                             name=name, agent_id=agent_id))

            # collect action outputs of all actors
            self.obs_phs = [actor.obs_tensor for actor in self.actors]
            self.act_phs = [actor.act_tensor for actor in self.actors]

            for agent_id in range(self.env.n):
                with tf.name_scope("critic_{}_{}".format(name, agent_id)):
                    act_phs = self._mask_other_act_phs(self.act_phs, agent_id)  # stop gradient
                    self.critics.append(Critic(self.sess, self.obs_phs, act_phs, lr=critic_lr, name=name, agent_id=agent_id))
                    self.actions_dims.append(self.env.action_space[agent_id].n)

            # set optimizer for actors
            for actor, critic in zip(self.actors, self.critics):
                with tf.name_scope("optimizer_{}_{}".format(name, agent_id)):
                    actor.set_optimization(critic)

    def init(self):
        self.sess.run(tf.global_variables_initializer())

        # hard sync
        for i in range(self.env.n):
            self.actors[i].update()
            self.critics[i].update()

    @staticmethod
    def _mask_other_act_phs(act_phs, agent_id):
        """ Mask action placeholders corresponding to other agents whose id != agent_id

        :param act_phs: list, action placeholder list
        :param agent_id: int, agent id
        :return: list of masked action placeholder
        """

        res = []
        for i, ph in enumerate(act_phs):
            if agent_id == i:
                res.append(ph)
            else:
                res.append(tf.stop_gradient(ph))

        return res

    def store_trans(self, state_n, action_n, next_state_n, reward_n, done_n):
        self.replay_buffer.push(state_n, action_n, next_state_n, reward_n, done_n)

    def act(self, obs_set, noise=0.0):
        """ Accept a observation list, return action list of all agents. """
        actions = []
        for i, (obs, agent) in enumerate(zip(obs_set, self.actors)):
            n = self.actions_dims[i]
            actions.append(agent.act(obs) + np.random.randn(n) * noise)
        return actions

    def async_update(self):
        """ Soft update target actors and critics. """

        for j in range(self.env.n):
            self.actors[j].soft_update()
            self.critics[j].soft_update()

    def save(self, dir_path, epoch):
        """ Save model
        :param dir_path: str, the grandparent directory which stores all models
        :param epoch: int, number of current round
        """

        dir_name = os.path.join(dir_path, self.name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        saver = tf.train.Saver(model_vars)
        save_path = saver.save(self.sess, dir_name + "/{}_model_{}.ckpt".format(self.name, epoch))
        print("[*] Model saved in file: {}".format(save_path))

    def load(self, dir_path, epoch=0):
        """ Load model from local storage, if no such model file, it will print warning to you

        :param dir_path: str, the grandparent directory which stores all models
        :param epoch: int, the index which used for indicating a certain model file
        """

        file_path = None

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
        """ Train actor and critic step by step

        :param batch_list: list, list of batch data for N agents
        :return: a_loss, c_loss: two lists for N agents which are actor-loss and critic-loss respectively
        """
        a_loss, c_loss = [0.] * self.n_agent, [0.] * self.n_agent

        state_clus = [None for _ in range(self.n_agent)]
        next_state_clus = [None for _ in range(self.n_agent)]
        act_clus = [None for _ in range(self.n_agent)]

        # re-construct samples
        for i, batch in enumerate(batch_list):
            state_clus[i] = np.concatenate(batch.state, axis=1)
            next_state_clus[i] = np.concatenate(batch.next_state, axis=1)
            act_clus[i] = np.concatenate(batch.action, axis=1)

        # get target action
        next_act_clus = [None for _ in range(self.n_agent)]
        for i, batch in enumerate(batch_list):
            next_act_clus[i].append(self.actors[i].target_act(batch.next_state))

        # train critic
        for i, batch in enumerate(batch_list):
            target_q = self.critics[i].calculate_target_q(next_state_clus, next_act_clus).reshape((-1,))
            target_q = np.array(batch.reward) + (1. - np.array(batch.done)) * target_q * self.gamma

            c_loss[i] = self.critics[i].train(target_q.reshape((-1, 1)), state_clus, act_clus)

        # train actor
        for i, batch in enumerate(batch_list):
            a_loss[i] = self.actors[i].train(batch.state)

        return a_loss, c_loss

    def train(self):
        """ Run multiple training steps

        :return: mean_a_loss, mean_c_loss: mean of actor loss and critic loss for N agents
        """

        n_batch = 1
        mean_a_loss, mean_c_loss = [0.] * self.n_agent, [0.] * self.n_agent

        for i in range(n_batch):
            batch_list = self.replay_buffer.sample(self.batch_size)
            a_loss, c_loss = self.train_step(batch_list)

            mean_a_loss = map(operator.add, mean_a_loss, a_loss)
            mean_c_loss = map(operator.add, mean_c_loss, c_loss)

        mean_a_loss = list(map(lambda x: x / n_batch, mean_a_loss))
        mean_c_loss = list(map(lambda x: x / n_batch, mean_c_loss))

        return mean_a_loss, mean_c_loss
