mport numpy as np
import tensorflow as tf


class BaseModel(object):
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.params = kwargs
        self.sess = None
        self.loss = None
        self.train_data = None

        # === define base info ===
        self.obs = tf.placeholder(tf.float32, shape=(None, self.params["obs_space"]), name="obs_input")
        self.action_labels = tf.placeholder(tf.float32, shape=(None, self.params["action_space"]), name="policy_output")

    def constrct_nn(self, active_func, name="default"):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def dump(self):
        pass

    def store(self):
        pass


class SuperVised(BaseModel):
    def __init__(self, name, **kwargs):
        super().__init__(name, kwargs)

        # === construct network ===
        with tf.get_variable_scope(self.name):
            self.policy = self.construct()

        self.loss = tf.losses.sigmoid_cross_entropy(self.action_labels, self.policy, weights=self.params["batch_size"])
        self.train_op = tf.train.AdamOptimizer(self.params["learning_rate"]).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def construct(self, active_func, name="default"):
        with tf.get_variable_scope(name):
            layer1 = tf.layers.dense(self.obs, units=256, activation=active_func)
            layer2 = tf.layers.dense(layer1, units=128, activation=active_func)
            layer3 = tf.layers.dense(layer2, units=64, activation=active_func)
            out = tf.layers.dense(layer3, units=self.params["action_space"])
        return out

    def train(self, data):
        # mini-batch
        data_obs, data_actions = data["obs"], data["actions"]
        length = data_obs.shape[0]
        batch_num = (length + self.params["batch_size"] - 1) // self.params["batch_size"]
        loss_seq, losses = [], 0.0
        print_every = self.params["print_every"]
        logger = self.params["logger"]

        for i in range(self.params["iteration"]):
            losses = 0.0
            for j in range(batch_num):
                idx = np.random.choice(length, size=batch_size, replace=False)
                batch_obs = data_obs[idx]
                batch_labels = data_actions[idx]
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                    self.obs: batch_obs,
                    self.action_labels: batch_labels})
                losses += loss
            
            loss_seq.append(losses / batch_num)
            if i % print_every == 0:
                mess = "[iteration {0}, loss]\t{1}".format(i, losses)
                print(mess)
                logger.info(mess)

        if self.params["store"]:
            file_path = "{0}/{1}.cpkt".format(self.params["path"], self.name)
            self.dump(file_path)
            print("--- DONE: Model has been stored as {} ---".format(file_path))


class Rinforcement(BaseModel):
    def __init__(self, name, **kwargs):
        super().__init__(name, kwargs)

        self.reward = tf.placeholder(tf.float32, shape=(None, 1))  # for target Q-value calculation
        self.obs_next = tf.placeholder(tf.float32, shape=(None, self.params["obs_space"]))
        
        # == construct network
        with tf.get_variable_scope(self.name):
            self.q_eval = self.constrct_nn(self, tf.nn.relu, "eval_net")
            self.q_target = self.constrct_nn(self, tf.nn.relu, "target_net")

        # == loss and training operation ==
        self.loss = tf.reduce_mean(tf.square(self.reward + self.params["gamma"] * self.q_target - self.q_eval))
        self.train_op = tf.train.AdamOptimizer(self.params["learning_rate"]).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def constrct_nn(self, activation, name="default"):
        with tf.get_variable_scope(name):
            layer1 = tf.layers.dense(self.obs, units=256, activation=active_func)
            layer2 = tf.layers.dense(layer1, units=128, activation=active_func)
            layer3 = tf.layers.dense(layer2, units=64, activation=active_func)
            out = tf.layers.dense(layer3, units=self.params["action_space"])

        # select the maximum of all Q-value from the last layer
        max_q = tf.reduce_max(out)
        return max_q

    def train(self):
        # mini-batch
        # orginal data constains: (current_obs, action, reward, current_next)
        data_obs, _, data_rewards, data_obs_target  = data
        length = data_obs.shape[0]

        batch_num = (length + self.params["batch_size"] - 1) // self.params["batch_size"]
        loss_seq, losses = [], 0.0
        print_every = self.params["print_every"]
        logger = self.params["logger"]

        for i in range(self.params["iteration"]):
            losses = 0.0
            for j in range(batch_num):
                # get mini-batch data for current training
                idx = np.random.choice(length, size=batch_size, replace=False)
                batch_obs, batch_obs_next = data_obs[idx], data_obs_target[idx]
                batch_reward = data_rewards[idx]

                # run
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                    self.obs: batch_obs,
                    self.obs_next: batch_obs_next,
                    self.action_labels: batch_labels,
                    self.reward: batch_reward})

                losses += loss
            
            loss_seq.append(losses / batch_num)
            if i % print_every == 0:
                mess = "[iteration {0}, loss]\t{1}".format(i, losses)
                print(mess)
                logger.info(mess)

        if self.params["store"]:
            file_path = "{0}/{1}.cpkt".format(self.params["path"], self.name)
            self.dump(file_path)
            print("--- DONE: Model has been stored as {} ---".format(file_path))

