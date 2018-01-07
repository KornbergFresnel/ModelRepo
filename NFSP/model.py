import random
import gym
import time
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
    
    def sample(self):
        """Random sample batch data from inner data
        """
        batch_data = dict(obs=None, reward=None, action=None, obs_next=None, done=None)

        # TODO: detail implementation

        return batch_data


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

    def _greedy_policy(self, obs, train=True):

        # === inner func: eps decay rule
        def eps_rule():
            """Eps rule return the max epsilon vlaue according to the global configuration.
            
            RULE: eps_range * {percentage of training step with decay}
            PERCENTAGE: max(0, (eps_count - train_step + start_step)) / eps_step_count
            """

            assert self.esp_step_count > 1

            return self.eps_low + self.eps_range * max(0, (self.eps_count - self.train_step + self.start_step)) / self.eps_step_count
        eps = eps_rule()
        
        if random.random() < eps:
            action = self.q_action.eval({self.eval_obs: obs})
        else:
            action = random.randint(self.env.action_spac.n)

        return action

    def pick_action(self, obs, policy='greedy', train=True):
        # run eval network
        if policy == "greedy":
            return self._greedy_policy(obs, train)
        else:
            return self.env.action_space.sample()
            
    def perceive(self, obs, action, reward, obs_next, done):
        # TODO: add current data to replay buffer, then training with mini-batch
        self.replaybuffer.put(obs, action, reward, obs_next, done)

        if self.train_step % self.update_every == self.update_every - 1:
            self._update_network()
        else:
            self._train()

    def _build_network(self):
        """This method implements the structure of DQN or DDQN,
        and for convenient, all weights and bias will be recorded in `self.e_w` and `self.t_w`
        """

        self.e_w = {}  # weight matrix for evaluation-network
        self.t_w = {}  # weight matrix for target-network
        
        init_func = tf.truncated_normal_initializer(0, 0.02)
        activation_func = tf.nn.relu

        # === Build Evaluation Network ===
        with tf.get_variable_scope("eval"):
            self.eval_obs = tf.placeholder(tf.float32, shape=(None, self.env.obs_space.shape(),), name="input_layer")
            self.l1, self.e_w["l1"], self.e_w["l1_b"] = ops.conv2d(self.eval_obs, 32, [8, 8], [4, 4], self.data_format, init_func, activation_func, name="l1")
            self.l2, self.e_w["l2"], self.e_w["l2_b"] = ops.conv2d(self.l1, 64, [4, 4], [2, 2], self.data_format, init_func,
                                                                   activation_func, name="l2")
            self.l3, self.e_w["l3"], self.e_w["l3_b"] = ops.conv2d(self.l2, 64, [3, 3], [1, 1], self.data_format, init_func,
                                                                   activation_func, name="l3")

            self.flat = tf.layers.flatten(self.l3)

            if self.dueling:
                pass
            else:
                # dense layer
                self.l4, self.e_w["l4"] ,self.e_w["l4_b"] = ops.custom_dense(self.flat, 512, activation_func, init_func, "dense_layer")
                self.e_q, self.e_w["q_w"], self.e_w["q_b"] = ops.custom_dense(self.l4, self.env.action_space.shape(), activation_func, init_func, "q_layer")

            self.q_action = tf.argmax(self.e_q, axis=1)  # record the index of final-layer, also map to the action index
        
        # === Build Target Network ===
        with tf.get_variable_scope("target"):
            self.target_obs = tf.placeholder(tf.float32, shape=(None, self.env.obs_space.shape(),), name="input_layer")
            self.t_l1, self.t_w["l1"], self.t_w["l1_b"] = ops.conv2d(self.target_obs, 32, [8, 8], [4, 4], self.data_format, init_func,
                                                                     activation_func, name="l1")
            self.t_l2, self.t_w["l2"], self.t_w["l2_b"] = ops.conv2d(self.t_l1, 64, [4, 4], [2, 2], self.data_format, init_func,
                                                                     activation_func, name="l2")
            self.t_l3, self.t_w["l3"], self.t_w["l3_b"] = ops.conv2d(self.t_l2, 64, [3, 3], [1, 1], self.data_format, init_func,
                                                                     activation_func, name="l3")

            self.t_flat = tf.layers.flatten(self.t_l3)

            if self.dueling:
                pass
            else:
                # dense layer
                self.t_l4, self.t_w["l4"] ,self.t_w["l4_b"] = ops.custom_dense(self.t_flat, 512, activation_func, init_func, "dense_layer")
                self.t_q, self.t_w["q_w"], self.t_w["q_b"] = ops.custom_dense(self.t_l4, self.env.action_space.shape(), activation_func, init_func, "q_layer")

                # if we training with double DQN, then the target network will produce an action with indicator from
                # evaluation network so the Q selection should accept an `index` tensor which depends on the result of
                # evaluation-network's selection
                self.target_q_idx_input = tf.placeholder(tf.int32, shape=(None, None), name="DDQN_max_action_index")
                self.target_q_action_with_idx = tf.gather_nd(self.t_q, self.target_q_idx_input)
        
        # === Define the process of network update ===
        with tf.get_variable_update("update"):
            self.t_w_input = {}  # record all weights' input of target network
            self.t_w_assign_op = {}  # record all update operations

            for name in self.e_w.keys():
                self.t_w_input[name] = tf.placeholder(tf.float32, shape=self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])
        
        # === Define the optimization ===
        with tf.get_variable_scope("optimization"):
            self.t_q_input = tf.placeholder(tf.float32, shape=(None, self.env.action_space.shape()), name="target_q_input")
            self.action_input = tf.placeholder(tf.int32, shape=(None,), name="action_input")
            
            action_one_hot = tf.one_hot(self.action_input, self.env.action_space.shape(), on_value=1.0, off_value=0.0, name="action_one_hot")
            q_eval_with_act = tf.reduce_sum(self.e_q * action_one_hot, axis=1, name="q_eval_with_action")

            self.loss = 0.5 * tf.reduce_sum(tf.square(self.t_q_input - q_eval_with_act))

            # TODO: consider add variant leraning rate

            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
    
    def _update(self):
        """Implement the network update
        """

        length = len(self.t_w)

        for name in enumerate(self.t_w.keys()):
            self.t_w_assign_op[name].eval({self.t_w_input: self.e_w[name]})
            print("[*] ---> Network Update: {0:.3f}%".format(i / length * 100.0))

    def _train(self):
        """Execute the training task with `mini_batch` setting.
        and this traninig module will training with game emulator"""

        self.num_train += 1
        print("[*] Begin {}th training ...".format(self.num_train))

        loss = []
        start_time = time.time()

        for _ in tqdm(range(self.iteration), cols=50):
            # emulator for training
            info = self._mini_batch()
            loss.append(info.loss)
        
        end_time = time.time()

        print("[*] Time consumption: {0:.3f}s, Average loss: {1}".format(end_time - start_time, sum(loss) / len(loss)))
        
    def _mini_batch(self):
        """Implement mini-batch training
        """

        info = dict(loss=0.0, time_consumption=0.0)  # info registion

        # sample from replay-buffer
        data_batch = self.replaybuffer.sample()

        if self.use_double:
            pred_act_batch = self.q_action.eval({self.eval_obs: data_batch.obs_next})  # get the action of next observation
            q_value_with_max_idx = self.target_q_action_with_idx.eval({
                self.target_obs: data_batch.obs_next,
                self.t_q_input: [[idx, act_idx] for idx, act_idx in enumerate(pred_act_batch)]
            })
            target_q = (1. - data_batch.done) * data_batch.reward + q_value_with_max_idx
        else:
            q_value = self.t_q.eval({self.target_obs: data_batch.obs_next})
            max_q_value = np.max(q_value, axis=1)
            target_q = (1. - data_batch.done) * data_batch.reward + max_q_value
        
        loss = self.train_op.eval({
            self.t_q_input: target_q,
            self.action_input: data_batch.action,
            # self.learning_rate_step: self.train_step
        })

        return info

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

        if (episode + 1) % 100 == 0:  # test every 100 episodes
            print("[*] === Enter TEST module ===")
            test(env, agent)


if __name__ == "__main__":
    main()
