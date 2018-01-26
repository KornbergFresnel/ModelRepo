import random
import time
import tqdm
import tensorflow as tf
import numpy as np

from base import ReplayBuffer, BaseModel, SReplayBuffer


class DQN(BaseModel):
    def __init__(self, env, config):
        super(DQN, self).__init__("dqn", config)
        
        self.env = env
        self.batch_size = config.batch_size
        self.replay_buffer = ReplayBuffer(config.batch_size, config.memory_size, env.observation_space.shape)
        self.num_train = 0

        self.input_layer = tf.placeholder(tf.float32, (None,) + self.env.observation_space.shape)

        with tf.variable_scope(self.name):
            self._build_network()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

        self._saver = tf.train.Saver(model_vars)

    def _greedy_policy(self, obs):
        if random.random() < self.eps:
            with self.sess.as_default():
                action = self.q_action.eval({self.input_layer: obs.reshape((1, 4))})[0]
        else:
            action = self.env.action_space.sample()

        return action

    def pick_action(self, obs, policy='greedy', train=True):
        # run eval network
        if train:
            if policy == "greedy":
                return self._greedy_policy(obs)
            else:
                return self.env.action_space.sample()
        else:
            with self.sess.as_default():
                action = self.q_action.eval({self.input_layer: obs.reshape((1, 4))})[0]
            return action
            
    def perceive(self, obs, action, reward, done):
        self.replay_buffer.put(obs, action, reward, done)

    def train(self, num_train):
        self._train(num_train)
        self.eps = max(self.eps_decay, (self.eps - self.eps_decay))

    def _build_network(self):
        """This method implements the structure of DQN or DDQN,
        and for convenient, all weights and bias will be recorded in `self.e_w` and `self.t_w`
        """
        activation_func = tf.nn.relu

        # === Build Evaluation Network ===
        with tf.variable_scope("eval"):
            self.eval_scope_name = tf.get_variable_scope().name

            self.l1 = tf.layers.dense(self.input_layer, units=20, activation=activation_func, name="eval_l1")
            self.l2 = tf.layers.dense(self.l1, units=20, activation=activation_func, name="eval_l2")
            self.l3 = tf.layers.dense(self.l2, units=20, activation=activation_func, name="eval_l3")

            if self.dueling:
                pass
            else:
                # dense layer
                self.e_q = tf.layers.dense(self.l3, units=self.env.action_space.n, activation=activation_func,
                                           use_bias=False, name="eval_q")

            # record the index of final-layer, also map to the action index
            self.q_action = tf.argmax(self.e_q, axis=1, name="eval_action_select")
        
        # === Build Target Network ===
        with tf.variable_scope("target"):
            self.target_scope_name = tf.get_variable_scope().name

            self.t_l1 = tf.layers.dense(self.input_layer, units=20, activation=activation_func, name="target_l1")
            self.t_l2 = tf.layers.dense(self.t_l1, units=20, activation=activation_func, name="target_l2")
            self.t_l3 = tf.layers.dense(self.t_l2, units=20, activation=activation_func, name="target_l3")

            if self.dueling:
                pass
            else:
                # dense layer
                self.t_q = tf.layers.dense(self.t_l3, units=self.env.action_space.n, activation=activation_func,
                                           use_bias=False, name="target_q")

                # if we training with double DQN, then the target network will produce an action with indicator from
                # evaluation network so the Q selection should accept an `index` tensor which depends on the result of
                # evaluation-network's selection
            self.target_q_idx_input = tf.placeholder(tf.int32, shape=(None, None), name="DDQN_max_action_index")
            self.target_q_action_with_idx = tf.gather_nd(self.t_q, self.target_q_idx_input)

        # === Define the process of network update ===
        with tf.variable_scope("update"):
            self.update_op = []

            eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.eval_scope_name)
            target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.target_scope_name)

            for i in range(len(target_params)):
                self.update_op.append(tf.assign(target_params[i], eval_params[i]))
        
        # === Define the optimization ===
        with tf.variable_scope("optimization"):
            self.t_q_input = tf.placeholder(tf.float32, shape=(None,), name="target_q_input")
            self.action_input = tf.placeholder(tf.int32, shape=(None,), name="action_input")
            
            action_one_hot = tf.one_hot(self.action_input, self.env.action_space.n, on_value=1.0, off_value=0.0, name="action_one_hot")
            self.q_eval_with_act = tf.reduce_sum(self.e_q * action_one_hot, axis=1, name="q_eval_with_action")

            temp = tf.square(self.t_q_input - self.q_eval_with_act)
            self.loss = 0.5 * tf.reduce_mean(temp)

            # TODO: consider add variant leraning rate

            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
    
    def _update(self):
        """Implement the network update
        """
        self.sess.run(self.update_op)

    def _train(self, num_train):
        """Execute the training task with `mini_batch` setting.
        and this traninig module will training with game emulator"""

        print("\n[*] Begin #{0} training / EPS: {1:.3f} / MemorySize: {2} ...".format(num_train, self.eps, self.replay_buffer.size)
        time.sleep(0.5)

        loss = []
        target_q_value = []
        eval_q_value = []
        start_time = time.time()

        buffer_size = self.replay_buffer.size
        self.iteration = (buffer_size + self.batch_size - 1) // self.batch_size

        for i in tqdm.tqdm(range(self.iteration), ncols=60):
            # emulator for training
            info = self._mini_batch()
            loss.append(info["loss"])
            target_q_value.append(info["target_q"])
            eval_q_value.append(info["eval_q"])
            if (i + 1) % self.update_every == 0:
                self._update()

        end_time = time.time()
        time.sleep(0.01)

        # loss record
        mean_loss = sum(loss) / len(loss)
        max_q, min_q = max(target_q_value[-1]), min(target_q_value[-1])
        max_e, min_e = max(eval_q_value[-1]), min(eval_q_value[-1])

        self.loss_record.append(mean_loss)

        print("\n[*] Time consumption: {0:.3f}s, Average loss: {1:.6f}, Max-q: {2:.6f}, Min-q: {3:.6f}, Max-e: {4:.6f}, Min-e: {5:.6f}"
              .format(end_time - start_time, mean_loss, max_q, min_q, max_e, min_e))
        
    def _mini_batch(self):
        """Implement mini-batch training
        """

        info = dict(loss=0.0, time_consumption=0.0)  # info registion

        # sample from replay-buffer
        data_batch = self.replay_buffer.sample()

        with self.sess.as_default():
            if self.use_double:
                pred_act_batch = self.q_action.eval({self.input_layer: data_batch.obs_next})  # get the action of next observation
                max_q_value = self.target_q_action_with_idx.eval({
                    self.input_layer: data_batch.obs_next,
                    self.target_q_idx_input: [[idx, act_idx] for idx, act_idx in enumerate(pred_act_batch)]
                })
            else:
                t_q_value = self.t_q.eval({self.input_layer: data_batch.obs_next})
                max_q_value = np.max(t_q_value, axis=1)

        # target_q = (1. - data_batch.done) * max_q_value * self.eps + data_batch.reward
        target_q = np.where(data_batch.done, data_batch.reward, data_batch.reward + max_q_value * self.eps)
        
        info["loss"], info["eval_q"], _ = self.sess.run([self.loss, self.q_eval_with_act, self.train_op], {
            self.t_q_input: target_q,
            self.action_input: data_batch.action,
            self.input_layer: data_batch.obs
            # self.learning_rate_step: self.train_step
        })

        info["target_q"] = target_q

        return info


class SuperVised(BaseModel):
    """Implemented with supervised neural network, learning policy for agents
    """

    def __inist(self, env, config):
        super(SuperVised, self).__init__("supervisde", config)

        self._env = env
        self._batch_size = config.batch_size
        self._repaly_buffer = SReplayBuffer(config.batch_size, config.memory_size, 
                                            self._env.observation_space.shape, self._env.action_space.n)
        self._build_network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def _build_network(self):
        act_func = tf.nn.relu

        with tf.variable_scope("supervised"):
            self.input_layer = tf.placeholder(tf.float32, shape=(None,) + self._env.observation_space.shape)

            self.l1 = tf.layers.dense(self.input_layer, units=20, activation=act_func, name="l1")
            self.l2 = tf.layers.dense(self.l1, units=20, activation=act_func, name="l2")
            self.l3 = tf.layers.dense(self.l2, units=10, activation=act_func, name="l3")

            self.pred_policy = tf.layers.dense(self.l3, units=self._env.action_space.n, use_bias=False, name="pred_layer")
        
        with tf.variable_scope("optimization"):
            self.labels = tf.placeholder(tf.float32, shape=(None, self._env.action_space.n), name="labels")

            # Here, we use cross-entropy to calculate the training loss
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.pred_policy) 
            self.loss = tf.reduce_mean(diff)

            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
    
    def _mini_batch(self):
        data = self._repaly_buffer.sample()

        loss = self.sess.run([self.loss, self.train_op], feed_dict={
            self.input_layer: data.obs
            self.labels: data.actions
        })

        return loss
    
    def train(self, train_num):
        iter_time = (self.repaly_buffer.size + self._batch_size - 1) // self._batch_size

        print("[* Super] Begin #{} training".format(train_num))

        start_time = time.time()

        for i in range(iter_time):
            self.loss_record.append(self._mini_batch())
        
        end_time = time.time()

        print("[* Super] Time concumption {0:.3f}".format(end_time - start_time))
    
    def update_buffer(self, observation, action):
        self._repaly_buffer.put(observation, action)


class NFSP(object):
    def __init__(self, env, config):
        self.sub = dict(dqn=DQN(env, config), sup=SuperVised(env, config))
        self._eta = config.eta
    
    def train(self, train_num):
        # 1. Get policy or best-response from DQN
        # 2. Get policy approximation from SuperVised-Network
        # 3. Select policy between DQN and SuperVised-Network with eta probability
        prob = random.random(0, 1)

        if prob < self._eta:
            # Select best-response
            pass
        else:
            # Select policy approximation
            pass
