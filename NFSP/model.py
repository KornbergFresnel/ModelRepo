import time
import random
import numpy as np
import tensorflow as tf
from base import BaseModel
from lib.tools import ops
from tqdm import tqdm
from lib.utils import History, ReplayBuffer


class Reinforcement(BaseModel):
    def __init__(self, env, config):
        super().__init__(config)

        self.history = History(config)
        self.replay_buffer = ReplayBuffer(config)
        self.env = env

        self.total_loss = []

        self._construct_nn()

    @property
    def eps_rule(self):
        """Eps rule return the max epsilonvalue according to 
        the global configuration.

        RULE: eps_range * {percentage of traninig step with decay}
        PERCENTAGE: max(0, (eps_count - train_step + start_step)) / eps_step_count
        """
        assert self.eps_setp_count > 1

        return self.eps_low + self.eps_range * max(0, (self.eps_count - self.train_step + self.start_step)) / self.eps_step_count

    def _update_network(self):
        """Update target network with eval network.
        """

        print("[*] Update target network with eval network...")

        for key in self.t_w.keys():
            self.t_w_assign_op[key].eval({self.t_w_input[key]: self.e_w[key].eval()})

    def _mini_batch_training(self):
        """Implement mini-batch training
        """
        # sample
        obs_batch, reward_batch, action_batch, obs_next_batch, terminal_batch = \
                self.replay_buffer.sample()

        if self.use_double:
            pred_act_batch = self.q_action.eval({self.s_t: obs_next_batch})
            q_value_with_max_idx = self.target_q_with_idx.eval({
                self.target_s_t: obs_next_batch,
                self.target_q_idx: [[idx, act_idx] for idx, act_idx in enumerate(pred_act_batch)]
                })

            target_q = (1. - terminal_batch) * reward_batch + q_value_with_max_idx
        else:
            q_value = self.target_q.eval({self.target_s_t: obs_next_batch})
            max_q_value = np.max(q_value, axis=1)
            target_q = (1. - terminal_batch) * reward_batch + max_q_value

        print("[*] Start {0}/{1} training..".format(self.train_step, self.train_iter))

        start_t = time.time()

        loss = self.train_op.eval({
            self.target_q_input: target_q,
            self.action_input: action_batch,
            self.s_t: obs_batch,
            self.learning_rate_step: self.train_step
        })

        end_t = time.time()

        print("[*] --- time consumption: {0:.2f}, loss: {1} ---".format(end_t - start_t, loss))

        self.total_loss.append(loss)

    def _observe(self, obs, reward, action, terminal):
        """Produce new experiences and add them to buffer, then select
        training or update current model
        """

        # reward rectify
        reward = max(self.reward_min, min(self.reward_max, reward))

        # update history and replay buffer
        self.history.add(obs)
        self.replay_buffer.add(obs, reward, action, terminal)

        # train or update
        if self.train_step % self.update_every == self.update_every - 1:
            self._update_network()
        else:
            self._mini_batch_training()

    def _predict(self, obs, figure_eps=None):
        """Make action prediction accroding to a certain observation
        with default epsilon-greedy policy, and the epsilon obeys the
        decay rule which defined at global configuration.
        """

        eps = figure_eps or self.eps_rule

        if random.random() < eps:
            action = self.q_action.eval({self.s_t: obs})
        else:
            action = random.choice(self.env.action_size)

        return action

    def play(self, n_episode=100, n_steps=10000, test_ep=None, render=False):
        test_ep = test_ep or self.eps_low

        test_history = History(self.config)

        # if not self.display:
        #    pass

        best_total_reward, best_episode = 0., 0

        for epis in range(n_episode):
            obs, reward, action, terminal = self.env.new_random_game()
            current_reward = 0.

            for _ in range(self.history_length):
                test_history.append(obs)

            for t in tqdm(range(n_steps), ncols=50):
                action = self._predict(test_history.get(), test_eps)
                obs, reward, terminal = self.env.act(action)
                test_history.add(obs)

                current_reward += reward

                if terminal:
                    break

            if current_reward > best_total_reward:
                best_total_reward = current_reward
                best_episode = epis

            print("[*] Best total reward: {0} related episode: {1}.".format(best_total_reward, best_episode))

        if not self.display:
            self.env.env.monitor.close()

    def train(self):
        """Execute the training task with `mini_batch` setting.
        and this tranining module will training with game emulator
        """

        start_time = time.time()

        obs, reward, action, terminal = self.env.new_random_game()

        for self.train_step in range(self.history_length):
            self.history.add(obs)

        for _ in tqdm(range(0, self.train_iter), ncols=50):
            # always update the history which used for update replay buffer
            action = self.predict(self.history.get())
            obs_next, reward, terminal = self.env.act(action)

            self._observe(obs_next, reward, terminal)

            # TODO: finish work for traninig, such as make a summary ?
        end_time = time.time()

        print("[*] Traninig task ended with time consumption: {0:.2f}s".format(end_time - start_time))

    def _construct_nn(self):
        """This method implements the structure of DQN or DDQN,
        and for convinient, all weights and bias will be recorded in `self.e_w` and `self.t_w`,
        """

        self.e_w = {}  # for eval-net
        self.t_w = {}  # for target-net

        init_func = tf.truncated_normal_initializer(0, 0.02)  # using tuncated normal to overcome saturation of tome function like sigmoid
        activation_func = tf.nn.relu

        # === build evaluation network ===
        with tf.get_variable_scope("eval_net"):
            if self.data_format == "NHWC":
                self.s_t = tf.placeholder(tf.float32,\
                        shape=(None, self.obs_height, self.obs_width, self.history_length), name="obs_target") 
            elif self.data_format == "NCHW":
                self.s_t = tf.placeholder(tf.float32, \
                        shape=(None, self.history_length, self.obs_height, self.obs_width), name="obs_target")

            self.l1, self.e_w["l1_w"], self.e_w["l1_b"] = ops.conv2d(self.s_t, 32, [8, 8], [4, 4], \
                    self.data_format, init_func, activation_func, name="l1")
            self.l2, self.e_w["l2_w"], self.e_w["l2_b"] = ops.conv2d(self.l1, 64, [4, 4], [2, 2], \
                    self.data_format, init_func, activation_func, name="l2")
            self.l3, self.e_w["l3_w"], self.e_w["l3_b"] = ops.conv2d(self.l2, 64, [3, 3]. [1, 1], \
                    self.data_format, init_func, activation_func, name="l3")

            l3_shape = self.l3.get_shape().as_list()

            self.flat = tf.placeholder(tf.float32, shape=(-1, reduce(lambda x, y: x * y, l3_shape[1:])))

            if self.dueling:
                # TODO: implement dueling layer
                pass
            else:
                # dense layer
                self.l4, self.e_w["l4_w"], self.e_w["l4_b"] = ops.custom_dense(self.flat, 512, activation_func, \
                        init_func, "dense_layer")
                self.q, self.e_w["q_w"], self.e_w["q_b"] = ops.custom_dense(self.l4, self.env.action_size, activation_func, \
                        init_func, "q_layer")
            # greedy: get maximum of Q-value function
            self.q_action = tf.argmax(self.q, axis=1)

        # === build target network ===
        with tf.get_variable_scope("traget_net"):
            if self.data_format == "NHWC":
                self.target_s_t = tf.placeholder(tf.float32,\
                        shape=(None, self.obs_height, self.obs_width, self.history_length), name="obs_target") 
            elif self.data_format == "NCHW":
                self.target_s_t = tf.placeholder(tf.float32, \
                        shape=(None, self.history_length, self.obs_height, self.obs_width), name="obs_target")

            self.t_l1, self.t_w["la1_w"], self.t_w["l1_b"] = ops.conv2d(self.target_s_t, 32, [8, 8], [4, 4], \
                    self.data_format, init_func, activation_func, name="l1")
            self.t_l2, self.t_w["l2_w"], self.t_w["l2_b"] = ops.conv2d(self.t_l1, 64, [4, 4], [2, 2], \
                    self.data_format, init_func, activation_func, name="l2")
            self.t_l3, self.t_w["l3_w"], self.t_w["l3_b"] = ops.conv2d(self.t_l2, 64, [3, 3]. [1, 1], \
                    self.data_format, init_func, activation_func, name="l3")

            l3_shape = self.l3.get_shape().as_list()

            self.t_flat = tf.layers.flatten(self.t_l3, name="flatten")

            if self.dueling:
                # TODO: implement dueling layer
                pass
            else:
                # dense layer
                self.t_l4, self.t_w["l4_w"], self.t_w["l4_b"] = ops.custom_dense(self.t_flat, 512, activation_func, \
                        init_func, "dense_layer")
                self.target_q, self.t_w["q_w"], self.t_w["q_b"] = ops.custom_dense(self.t_l4, self.env.action_size, activation_func, \
                        init_func, "q_layer")

                # under double dqn, the target network should return target-q according to the idx which
                # from eval-net
                self.target_q_idx = tf.placeholder(tf.int32, shape=(None, None), name="ddqn_max_action_idx")
                self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        # === define network replacement operation
        with tf.get_variable_scope("update"):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.e_w.keys():
                self.t_w_input[name] = tf.placeholder(tf.float32, shape=self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        # === define optimizer: with learning-rate decay ===
        with tf.get_variable_scope("optimizer"):
            self.target_q_input = tf.placeholder(tf.float32, shape=(None,), name="target_q_t")
            self.action_input = tf.placeholder(tf.int32, shape=(None,), name="action")

            action_one_hot = tf.one_hot(self.action_input, self.env.action_size, on_value=1.0, off_value=0.0, name="action_one_hot")
            q_eval_with_act = tf.reduce_sum(self.q_eval * action_one_hot, axis=1, name="q_eval_with_action")

            self.loss = 0.5 * tf.reduce_mean(tf.square(self.target_q_input - q_eval_with_act))

            self.learning_rate_step = tf.placeholder(tf.int32, None, name="learning_rate_step")
            self.learning_rate_op = tf.maximum(self.learing_rate_min, \
                    tf.train.exponential_decay(self.learning_rate, self.learing_rate_step, \
                        self.learning_rate_decay_step, self.learning_rate_decay))

            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, \
                    epsilon=0.1).minimize(self.loss)

        # === define summary operation: for record
        # TODO: implement summary operation for training summary

        # === Finish work of construction ===
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self._saver = tf.train.Saver(self.e_w.values(), max_to_keep=self.max_to_keep)


class SuperVised(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self._construct_nn()
        self.memory = ReplayBuffer(config)
        self.total_loss = []

        # === traning record ===
        self.train_step = 0

    def play(self):
        pass

    def train(self):
        print("[* SuperVised] Start {0}/{1} training..".format(self.train_step, self.train_iter))

        start_time = time.time()

        for _ in tqdm(range(0, self.train_iter), ncols=50):
            # always update the history which used for update replay buffer
            obs_batch, _, action_batch, _, _ = self.replay_buffer.sample()
            loss = self.train_op.eval({self.s_t: obs_batch,
                self.label: action_batch})
            self.total_loss.append(loss)

            if self.train_step % self.print_every == (self.print_every - 1):
                print("[* SuperVised] loss: {1}, iter: {2}".format(loss))

        print("[* SuperVised] --- time consumption: {0:.2f}, loss: {1} ---".format(end_t - start_t, loss))

    def _construct_nn(self):
        init_func = tf.truncated_normal_initializer(0, 0.02)
        activation_func = tf.nn.relu

        if self.data_format == "NCHW":
            self.s_t = tf.placeholder(tf.float32, shape=(None, self.history_length, self.obs_height, self.width), name="super_input")
        elif self.data_format == "NHWC":
            self.s_t = tf.placeholder(tf.float32, shape=(None, self.obs_height, self.obs_width, self.history_length), name="super_input")

        # TODO: construct the supervised deep neural network
        with tf.get_variable_scope("supervised_nn"):
            self.l1 = ops.conv2d(self.s_t, 32, [8, 8], [4, 4],
                    self.data_format, init_func, activation_func, name="super_conv1")
            self.l2 = ops.conv2d(self.l1, 64, [4, 4], [2, 2],
                    self.data_format, init_func, activation_func, name="super_conv2")
            self.l3 = ops.conv2d(self.l2, 64, [3, 3], [1, 1],
                    self.data_format, init_func, activation_func, name="super_conv3")
            self.flatten = tf.layers.flatten(self.l3, name="flatten")
            self.dense = tf.layers.dense(self.flatten, 256, activation_func, name="dense")
            self.out = tf.layers.dense(self.dense, self.end.action_size, activation_func, False, name="outlayer")

        # === define optimizer layer ===
        with tf.get_variable_scope("optimizer"):
            self.loss = 0.5 * tf.reduce_mean(tf.square(self.out - self.label))
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())


class NFSP:
    def __init__(self, config):
        """This class combine both RL and SuperVised learning model, with sync traning and rendering
        """
        self.rl = Reinforcement(config)
        self.supervised = SuperVised(config)

    def train(self):
        # get action or policy from RL
        # get action or policy from supervised learning model
        self.supervised.
