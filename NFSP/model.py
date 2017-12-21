import numpy as np
import tensorflow as tf
from lib.base import BaseModel


class SuperVised(BaseModel):
    def __init__(self, name, **kwargs):
        super().__init__(name, kwargs)

        # === define a network ===
        self.obs = tf.placeholder(tf.float32, shape=(None, self.params["obs_space"]), name="obs_input")
        self.action_labels = tf.placeholder(tf.float32, shape=(None, self.params["action_space"]), name="policy_output")

        # === construct network ===
        self.policy = self.construct()
        self.loss = tf.losses.sigmoid_cross_entropy(self.action_labels, self.policy, weights=self.params["batch_size"])
        self.train_op = tf.train.AdamOptimizer(self.params["learning_rate"]).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def construct(self, active_func):
        with tf.get_variable_scope(self.name):
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

        # === define network ===
        self.obs = tf.placeholder(tf.float32, shape=(None, ))
