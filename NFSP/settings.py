config = {
        # === data structure description ===
        "data_format": "NCHW",
        "history_length": 4,
        "memory_size": 1024,

        # === training settings ===
        "batch_size": 64,  # control both batch size and sample buffer size
        "train_iter": 5000,
        "max_to_keep": 30,

        # === environment settings ===
        "env_name": None,
        "obs_height": 768,
        "obs_width": 1024,
        "display": True,
        "action_repeat": 4
    }
