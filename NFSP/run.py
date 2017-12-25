import numpy as np
from model import Supervised, Reinforcement
from lib.env import BlackjackEnv


def policy(model, env, eps=0.9):
    obs, done, _ = env.get_obs()
    Q = model.eval(obs)
    A = np.arange(2)
    a = np.random.choice()

