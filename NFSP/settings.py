class NetWorkConfig(object):
    """Configuration for both RL network and SuperVised network
    """

    scale = 10000
    display = False

    max_step = 5000 * scale
    memory_size = 100 * scale
    
    batch_size = 32
    random_start = 30
    data_format = "NCHW"
    
    train_iter = 2000
    max_to_keep = 30
    eps_low = 0.05
    eps_high = 1.0
    eps_count = 1000
    use_double = False
    dueling = False
    
    history_length = 4
    learning_rate_min = 1e-3
    learning_rate_max = 5e-3
    learning_rate_decay_step = 0


class EnvConfig(object):
    """Configuration of environment
    """

    env_name = "Breakout-v0"
    obs_height = 768
    obs_width = 1024
    display = False
    action_repeat= 4
    reward_min = -1
    reward_max = 1


class NFSPConfig(NetWorkConfig, EnvConfig):
    """Configration of NFSP
    """

    model = ""
    pass


class M1(NFSPConfig):
    backend = "tf"
    env_type = "detail"
    action_repeat = 1


class M2(NFSPConfig):
    backend = "tf"
    env_type = "detail"
    action_repeat = 4


def get_config(FLAGS):
    """Accept outer configuration then examplify current configuration
    """

    if FLAGS.model == "m1":
        config = M1
    elif FLAGS.model == "m2":
        config = M2

    for k, v in FLAGS.__dict__["__flags"].items():
        if k == "gpu":
            if v == False:
                config.data_format = "NHWC"
            else:
                config.data_format = "NCHW"

        if hasattr(config, k):
            setattr(config, k, v)
    
