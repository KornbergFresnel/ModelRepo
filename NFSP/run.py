import random
import tensorflow as tf
import numpy as np
from settings import get_config
from model import Reinforcement, SuperVised
from env.gymenvs import GymEnvironment, SimpleGymEnvironment


flags = tf.app.flags

# === Configuration: Model ===
flags.DEFINE_string("model", "m1", "Type of model")
flags.DEFINE_boolean("dueling", False, "Whether to use dueling DQN(DDQN)")
flags.DEFINE_boolean("use_double", False, "Whether to use DDQN")

# === Configuration: Environment ===
flags.DEFINE_string("env_name", "Breakout-v0", "The name of gym environment to use")
flags.DEFINE_integer("action_repeat", 4, "The number of action to be repeated")

# === Configuration: Traning & Testing===
flags.DEFINE_boolean("use_gpu", False, "Whether to use gpu or not")
flags.DEFINE_string("gpu_fraction", "1/1", "If you `use_gpu`, this argument will work, idx/# of gpu fraction e.g. 1/3, 2/3, 3/3")
flags.DEFINE_boolean("display", False, "Whether to display the game screen or not")
flags.DEFINE_boolean("is_train", True, "Whether traning or testing")
flags.DEFINE_integer("random_seed", 123, "Value of random seed")

FLAGS = flags.FLAGS

# === Configuration: Random seed ===
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)


if FLAGS.use_gpu and FLAGS.gpu_fraction == "":
    raise ValueError("--gpu_fraction should be defined")


def calc_gpu_fraction(fraction_str):
    """Calculate the GPU-fraction
    """

    idx, num = map(float, fraction_str.split("/"))

    fraction = 1 / (num - idx + 1)
    print("[*] GPU: {0:.4f}".format(fraction))

    return fraction


def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))

    with tf.Session(config=tf.ConfigProto(gpu_options)) as sess:
        config = get_config(FLAGS) or FLAGS
        
        if config.env_type == "simple":
            env = SimpleGymEnvironment(config)
        else:
            env = GymEnvironment(config)

        if not tf.test.is_gpu_available() and FLAGS.use_gpu:
            raise Exception("use_gpu flag is true when no GPUs are available")

        if not FLAGS.use_gpu:
            config.data_format = "NHWC"

        rl = Reinforcement(config)
        supervised = SuperVised(config)

        if FLAGS.is_train:
            rl.train()
            supervised.train()
        else:
            rl.play()
            supervised.play()


if __name__ == "__main__":
    tf.app.run()
    
