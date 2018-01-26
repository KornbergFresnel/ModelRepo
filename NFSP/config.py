class AgentConfig(object):
    use_double = True
    use_dueling = False
    data_format = "panel"


class TrainingConfig(object):
    memory_size = 1024  # for replay-buffer
    batch_size = 64  # for mini-batch

    # === configuration for training ===
    iteration = 512

    # === configuration for epsilon-greedy ===
    eps_decay = 0.002
    eps_high = 1.0

    # === configuration for learning rate decay ===
    learning_rate = 1e-4
    learning_decay = 1e-5

    # === configuration for saving ===
    max_to_keep = 10


class GeneralConfig(AgentConfig, TrainingConfig):
    EPISODE = 500
    ENV_NAME = "CartPole-v0"
    STEP = 300

    SAVE_EVERY = 10
    UPDATE_EVERY = 5
