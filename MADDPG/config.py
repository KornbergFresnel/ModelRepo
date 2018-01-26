import tensorflow as tf


class GeneralConfig():
    layers = [150, 150]
    batch_size = 64
    memory_size = 4096
    update_decay = 1.0
    temperature = 0.5

    critic_lr = 1e-3
    gamma = 0.99
    L2 = 0.01   
    actor_lr = 1e-4

    test_every = 5
    update_every = 5
    train_freq = 2

    log_dir = "./temp/maddpg"
    model_dir = "./models"