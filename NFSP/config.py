class AgentConfig(object):
    use_double = False
    use_dueling = False
    data_format = "panel"


class TrainingConfig(object):
    memory_size = 256
    batch_size = 64

    iteration = 512
    update_every = 5
    decay_start = 0
    decay_end = iteration

    eps_low = 0.05
    eps_high = 1.0
    eps_count = iteration

    learning_rate = 1e-3
    learning_decay = 1e-5

    max_to_keep = 10


class GeneralConfig(AgentConfig, TrainingConfig):
    test_every = 1
    pass
