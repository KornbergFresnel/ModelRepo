CONFIG = {
        # === data structure description ===
        "data_format": "NCHW",
        "history_length": 4,
        "memory_size": 1024,

        # === training settings ===
        "batch_size": 64,  # control both batch-size and buffer-size
        "train_iter": 2000,
        "max_to_keep": 30,
        "eps_low": 0.05,
        "eps_high": 1.0,
        "eps_count": 1000,
        "use_double": False,

        "learning_rate_min": 1e-3,
        "learning_rate_max": 5e-3,
        "learning_rate_decay_step": 0,

        # === environment settings ===
        "env_name": None,
        "obs_height": 768,
        "obs_width": 1024,
        "display": True,
        "action_repeat": 4,
        "reward_min": 0,
        "reward_max": 10
    }


def custom_config(**kwargs):
    for key in kwargs.keys():
        if CONFIG.get(key) is not None:
            CONFIG[key] = kwargs[key]
        else:
            print("[!] Error: `{}` is not a legally parameter.".format(key))
            exit(1)

