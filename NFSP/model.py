import time
import numpy as np
import tensorflow as tf
from base import BaseModel
from lib.tools import ops
from tqdm import tqdm
from lib.utils import History, ReplayBuffer


class Reinforcement(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.history = History(config)
        self.replay_buffer = ReplayBuffer(config)

    def train(self):
        """Execute the training task with `mini_batch` setting."""

        start_time = time.time()

        # TODO: init history buffer for training and traninig logic
        
        # for _ in tqdm(range(0, self.train_iter), ncols=50):  

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

            self.t_flat = tf.placeholder(tf.float32, shape=(-1, reduce(lambda x, y: x * y, l3_shape[1:])))

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
            self.target_q_t = tf.placeholder(tf.float32, shape=(None,), name="target_q_t")
            self.action = tf.placeholder(tf.int32, shape=(None,), name="action")

            action_one_hot = tf.one_hot(self.action, self.env.action_size, on_value=1.0, off_value=0.0, name="action_one_hot")
            q_eval_with_act = tf.reduce_sum(self.q_eval * action_one_hot, axis=1, name="q_eval_with_action")

            self.loss = 0.5 * tf.reduce_mean(tf.square(self.target_q_t - q_eval_with_act))

            self.learning_rate_step = tf.placeholder(tf.int32, None, name="learning_rate_step")
            self.learning_rate_op = tf.maximum(self.learing_rate_min, \
                    tf.train.exponential_decay(self.learning_rate, self.learing_rate_step, \
                        self.learning_rate_decay_step, self.learning_rate_decay))

            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, \
                    epislon=0.1).minimize(self.loss)

        # === define summary operation: for record
        # TODO: implement summary operation for training summary

        # === Finish work of construction ===
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self._saver = tf.train.Saver(self.e_w.values(), max_to_keep=self.max_to_keep)

