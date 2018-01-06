import random
import gym
import tqdm
import tensorflow as tf
import numpy as np

from collections import deque

from lib.tools import ops
from base import BaseModel


ENV_NAME = "CartPole-v0"
EPISODE = 10000
TEST_EPISODE = 300
STEP = 300  # step limitation


config = {
        "mini_batch": 64,
        "memory_size": 1024
    }


class ReplayBuffer(object):
    def __init__(self, **kwargs):
        # TODO: parse config
        self.flag = 0
        self.size = 0
        pass

    def push(self, obs, action, reward, obs_next, done):
        # TODO: push data to inner data buffer
        self.flag = (self.flag + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)
        pass

    def get(self):
        pass

    def free_memory(self):
        pass


class DQN(BaseModel):
    def __init__(self, env, config):
        self.env = env

        assert isinstance(config, dict)
        self.replaybuffer = ReplayBuffer(config)

        # TODO: parse config ( or kwargs )

        self._build_network()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver(self.e_w.values(), max_to_keep=self.max_to_keep)

    def pick_action(obs, policy='greedy', train=True):
        pass

    def perceive(obs, action, reward, obs_next, done):
        # TODO: add current data to replay buffer, then training with mini-batch
        self.replaybuffer.put(obs, action, reward, obs_next, done)

        if self.
        self._train()

    def _build_network(self):
        """This method implements the structure of DQN or DDQN,
        and for convinient, all weights and bias will be recorded in `self.e_w` and `self.t_w`
        """

        self.e_w = {}  # weight matrix for evaluation-network
        self.t_w = {}  # weight matrix for target-network
        
        init_func = tf.truncated_normal_initializer(0, 0.02)
        activation_func = tf.nn.relu

        with tf.get_variable_scope("eval"):
            self.eval_obs = tf.placeholder(tf.float32, shape=(None, self.env.obs_space.shape(),), name="input_layer")
            self.l1, self.e_w["l1"], self.e_w["l1_b"] = ops.conv2d(self.eval_obs, 32, [8, 8], [4, 4], self.data_format, init_func, \
                    activation_func, name="l1")
            self.l2, self.e_w["l2"], self.e_w["l2_b"] = ops.conv2d(self.l1, 64, [4, 4], [2, 2], self.data_format, init_func, \
                    activation_func, name="l2")
            self.l3, self.e_w["l3"], self.e_w["l3_b"] = ops.conv2d(self.l2, 64, [3, 3], [1, 1], self.data_format, init_func, \
                    activation_func, name="l3")

            self.flat = tf.layers.flatten(self.l3)

            if self.dueling:
                pass
            else:
                # dense layer
                self.l4, self.e_w["l4"] ,self.e_w["l4_b"] = ops.custom_dense(self.flat, 512, activation_func, init_func, "dense_layer")
                self.e_q, self.e_w["q_w"], self.e_w["q_b"] = ops.custom_dense(self.l4, self.env.action_space.shape(), activation_func, init_func, "q_layer")

            self.q_action = tf.argmax(self.e_q, axis=1)  # this tensor record the index of final-layer, also map to the action index

        with tf.get_variable_scope("target"):
            self.target_obs = tf.placeholder(tf.float32, shape=(None, self.env.obs_space.shape(),), name="input_layer")
            self.t_l1, self.t_w["l1"], self.t_w["l1_b"] = ops.conv2d(self.target_obs, 32, [8, 8], [4, 4], self.data_format, init_func, \
                    activation_func, name="l1")
            self.t_l2, self.t_w["l2"], self.t_w["l2_b"] = ops.conv2d(self.t_l1, 64, [4, 4], [2, 2], self.data_format, init_func, \
                    activation_func, name="l2")
            self.t_l3, self.t_w["l3"], self.t_w["l3_b"] = ops.conv2d(self.t_l2, 64, [3, 3], [1, 1], self.data_format, init_func, \
                    activation_func, name="l3")

            self.t_flat = tf.layers.flatten(self.t_l3)

            if self.dueling:
                pass
            else:
                # dense layer
                self.t_l4, self.t_w["l4"] ,self.t_w["l4_b"] = ops.custom_dense(self.t_flat, 512, activation_func, init_func, "dense_layer")
                self.t_q, self.t_w["q_w"], self.t_w["q_b"] = ops.custom_dense(self.t_l4, self.env.action_space.shape(), activation_func, init_func, "q_layer")

                # if we training with double DQN, then the target network will produce an action with indicator from evaluation network
                # so the Q selection should accept an `index` tensor which depends on the result of evalution-network's selection
                self.target_q_idx_input = tf.placeholder(tf.int32, shape=(None, None), name="ddqn_max_action_index")
                self.target_q_action_with_idx = tf.gather_nd(self.t_q, self.target_q_idx_input)

        with tf.get_variable_update("update"):
            self.t_w_input = {}  # record all weights' input of target network
            self.t_w_assign_op = {}  # record all update operations

            for name in self.e_w.keys():
                self.t_w_input[name] = tf.placeholder(tf.float32, shape=self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        with tf.get_variable_scope("optimization"):
            self.t_q_input = tf.placeholder(tf.float32, shape=(None, self.env.action_space.shape()), name="target_q_input")
            self.action_input = tf.placeholder(tf.int32, shape=(None,), name="action_input")
            
            action_one_hot = tf.one_hot(self.action_input, self.env.action_space.shape(), on_value=1.0, off_value=0.0, name="action_one_hot")
            q_eval_with_act = tf.reduce_sum(self.e_q * action_one_hot, axis=1, name="q_eval_with_action")

            self.loss = 0.5 * tf.reduce_sum(tf.square(self.t_q_input - q_eval_with_act))

            # TODO: consider add variant leraning rate

            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


    def _train(self):
        for i in range(self.iteration):
            

    def _mini_batch(self):
        pass


def test(env, agent):
    total_reward = 0

    for episode in range(TEST_EPISODE):
        obs = env.reset()
        done = False

        for _ in tqdm(range(STEP), ncols=50):
            # env.render()
            action = agent.pick_action(obs, train=False)
            obs, reward, done, _ = env.step(action)

            total_reward += reward

            if done:
                break

    ave_reward = total_reward / TEST_EPISODE
    mess = "[*] Test at episode: {0} with average reward: {1:.3f}".format(train_round, ave_reward)

    print(mess)


def main():
    env = gym.make(ENV_NAME)
    
    agent = DQN(env)

    for episode in range(EPISODE):

        obs = env.reset()

        # === Train ===
        for _ in tqdm(range(STEP), ncols=50):
            action = agent.pick_action(obs)
            obs_next, reward, done, _ = env.step(action)

            # agent will store the newest experience into replay buffer, and training with mini-batch and off-policy
            agent.perceive(obs, action, reward, obs_next, done)

            if done:
                break

            obs = obs_next

