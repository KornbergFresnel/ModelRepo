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

        self.learning_rate = config.actor_lr
        self.test_every = config.test_every
        self.decay = config.update_decay
        self.temperature = config.temperature

        self.sess = sess
        self.env = env
        self.agent_id = agent_id

        self.layers_conf = config.layers
        self.action_space = env.action_space[agent_id]
        self.observation_space = env.observation_space[agent_id]

        self.active_func = tf.nn.relu
        self.initialize = tf.truncated_normal_initializer(stddev=0.01)

        with tf.variable_scope("actor"):
            self.obs_input = tf.placeholder(tf.float32, shape=(None,) + self.observation_space.shape, name="Obs")
            with tf.variable_scope("eval"):
                self.eval_name_scope = tf.get_variable_scope().name
                self.e_out, self.e_variables = self._construct(self.action_space.n, self.eval_name_scope)
            
            with tf.variable_scope("target"):
                self.target_name_scope = tf.get_variable_scope().name
                self.t_out, self.t_variables = self._construct(self.action_space.n, self.eval_name_scope)

            with tf.variable_scope("Optimization"):
                self.q_gradient = tf.placeholder(tf.float32, shape=(None, self.action_space.n), name="Q-gradient")
                self.policy_gradients = tf.gradients(self.e_out, self.e_variables, self.q_gradient)
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                    zip(self.policy_gradients, self.e_variables)
                )

            with tf.variable_scope("Update"):  # smooth average update process
                self.update_op = [tf.assign(self.t_variables[i], self.decay * self.e_variables[i] + (1 - self.decay) * self.t_variables[i])
                                  for i in range(len(self.t_variables))]

    def _construct(self, out_dim, name_scope, norm=True):
        l1 = tf.layers.dense(self.obs_input, units=self.layers_conf[0], activation=self.active_func,
                             kernel_initializer=self.initialize, name="l1")
        if norm: l1 = tc.layers.layer_norm(l1)

        l2 = tf.layers.dense(l1, units=self.layers_conf[1], activation=self.active_func,
                             kernel_initializer=self.initialize, name="l2")
        if norm: l2 = tc.layers.layer_norm(l2)

        out = tf.layers.dense(l2, activation=tf.nn.softmax, units=out_dim, kernel_initializer=self.initialize)
        # out = tcd.RelaxedOneHotCategorical(self.temperature, probs=out).sample()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_scope)

        return out, variables

    def act(self, obs_set):
        obs_set = obs_set.reshape((1,) + obs_set.shape)
        policy = self.sess.run(self.e_out, feed_dict={self.obs_input: obs_set})
        return policy[0]

    def target_act(self, obs_set):
        """Return an action id -> integer"""
        policy = self.sess.run(self.e_out, feed_dict={self.obs_input: obs_set})
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

        self.learning_rate = config.critic_lr
        self.L2 = config.L2
        self.gamma = config.gamma
        self.layers_conf = config.layers
        self.update_every = config.update_every
        self.test_every = config.test_every
        self.decay = config.update_decay

        self.active_func = tf.nn.relu
        self.initial = tf.truncated_normal_initializer(stddev=0.01)

        with tf.variable_scope("critic"):
            self.mul_obs_input = tf.placeholder(tf.float32, shape=(None,) + self.mul_obs_dim, name="obs-input")
            self.mul_act_input = tf.placeholder(tf.float32, shape=(None,) + self.mul_act_dim, name="mul-act-input")
            self.input = tf.concat([self.mul_obs_input, self.mul_act_input], axis=1, name="concat-input")

            with tf.variable_scope("eval"):
                self.e_name_scope = tf.get_variable_scope().name
                self.e_q, self.e_variables = self._construct(self.e_name_scope)
            
            with tf.variable_scope("target"):
                self.t_name_scope = tf.get_variable_scope().name
                self.t_q, self.t_variables = self._construct(self.t_name_scope)
            
            with tf.variable_scope("Update"):  # smooth average update process
                self.update_op = [tf.assign(self.t_variables[i], self.decay * self.e_variables[i] + (1 - self.decay) * self.t_variables[i])
                                  for i in range(len(self.t_variables))]

            with tf.variable_scope("Optimization"):
                weight_decay = tf.add_n([self.L2 * tf.nn.l2_loss(var) for var in self.e_variables])
                self.t_q_input = tf.placeholder(tf.float32, shape=(None, 1), name="target-input")
                self.loss = 0.5 * tf.reduce_mean(tf.square(self.t_q_input - self.e_q)) + weight_decay
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                self.action_gradients = tf.gradients(-tf.reduce_mean(self.e_q), self.mul_act_input)

    def _construct(self, name_scope, norm=True):
        l1 = tf.layers.dense(self.input, units=self.layers_conf[0], activation=self.active_func,
                             kernel_initializer=self.initial, name="l1")
        if norm: l1 = tc.layers.layer_norm(l1)

        l2 = tf.layers.dense(l1, units=self.layers_conf[1], activation=self.active_func,
                             kernel_initializer=self.initial, name="l2")
        if norm: l2 = tc.layers.layer_norm(l2)

        q = tf.layers.dense(l2, units=1, kernel_initializer=self.initial, name="Q")
        
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name_scope)

        return q, variables
    
    def calculate_target_q(self, obs_next, action_next):
        q_values = self.sess.run(self.t_q, feed_dict={
            self.mul_obs_input: obs_next,
            self.mul_act_input: action_next
        })

        target_q_value = np.max(q_values, axis=1)
        
        return target_q_value * self.gamma

    def update(self):
        self.sess.run(self.update_op)
    
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
        self.record = None
        self.env = env

        self.actor = []  # hold all Actors
        self.critic = []  # hold all Critics
        self.actions_dims = []  # record the action split for gradient apply
        
        _CONFIG = GeneralConfig()
        self.update_every = _CONFIG.update_every
        self.train_freq = _CONFIG.train_freq


        # == Construct Network for Each Agent ==
        with tf.variable_scope(self.name):
            for agent_id in range(self.env.n):
                with tf.variable_scope(name + "_{}".format(agent_id)):
                    self.actor.append(Actor(env, self.sess, name, agent_id, _CONFIG))
                    self.critic.append(Critic(env, self.sess, name, agent_id, _CONFIG))

                    self.actions_dims.append(self.env.action_space[agent_id].n)
        
            # === Define summary ===
            self.reward = [tf.placeholder(tf.float32, shape=None, name="Agent_{}_reward".format(i)) for i in range(self.env.n)]
            self.reward_op = [tf.summary.scalar('Agent_{}_R_Mean'.format(i), self.reward[i]) for i in range(self.env.n)]

        self.sess.run(tf.global_variables_initializer())
        self.train_writer = tf.summary.FileWriter(_CONFIG.log_dir, graph=tf.get_default_graph())

        for i in range(self.env.n):
            self.actor[i].update()
            self.critic[i].update()
    
    def act(self, obs_set, noise=0.0):
        """Accept a observation list, return action list of all agents."""
        actions = []
        for i, obs in enumerate(obs_set):
            n = self.actions_dims[i]
            actions.append(self.actor[i].act(obs) + np.random.randn(n) * noise)
        return actions

    def summary(self, rewards, step):
        for i in range(self.env.n):
            self.train_writer.add_summary(self.sess.run(self.reward_op[i], feed_dict={self.reward[i]: np.mean(rewards[i])}), global_step=step)

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

    def update_nets(self):
        for j in range(self.env.n):
            self.actor[j].update()
            self.critic[j].update()

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

    def train(self, replay_buffer):
        """Training logical"""
        # n_buffer = len(replay_buffer)
        # n_batch = (n_buffer + replay_buffer.batch_size - 1) // replay_buffer.batch_size * self.train_freq
        n_batch = 1
        action_gradients = [[] for _ in range(self.env.n)]

        start = time.time()

        self.record = Record(self.env.n)

        # for i_batch in range(n_batch):
        batch_data = replay_buffer.sample()  # for all agents

        # data with `clus` prefix is prepared for critic
        obs_clus = np.concatenate(batch_data.obs, axis=1)
        obs_next_clus = np.concatenate(batch_data.obs_next, axis=1)
        action_clus = np.concatenate(batch_data.action, axis=1)
        batch_next_action = []

        for j in range(self.env.n):
            batch_next_action.append(self.actor[j].target_act(batch_data.obs_next[j]))

        batch_next_action = np.concatenate(batch_next_action, axis=1)

        for j in range(self.env.n):  # calculate Q-value for (s_next, a_next)
            batch_q_value = self.critic[j].calculate_target_q(obs_next_clus, batch_next_action)
            batch_q_value = batch_data.reward[:, j] + (1. - batch_data.terminate[:, j]) * batch_q_value

            # then training critic, return with eval-Q & loss
            critic_loss = self.critic[j].train(batch_q_value, obs_clus, action_clus)
            # compute action_gradients
            action_gradients[j], mean_q = self.critic[j].get_action_graidents(obs_next_clus, batch_next_action)

            self.record.reward[j] += mean_q
            self.record.loss[j] += critic_loss
        
        # reshape action_gradients for each agents
        action_gradients = self.restructure(np.array(action_gradients))

        for j in range(self.env.n):
            # then training actor: action_gradients should come from other agents, not only one
            self.actor[j].train(batch_data.obs_next[j], action_gradients[j])

            # if (i_batch + 1) % self.update_every:  # update and make summaries
            #     for j in range(self.env.n):
            #         self.actor[j].update()
            #         self.critic[j].update()

        end = time.time()
        total_reward = np.around(np.array(self.record.reward) / n_batch, decimals=6)
        mean_loss = np.around(np.array(self.record.loss) / n_batch, decimals=6)

        return mean_loss, total_reward, end - start
