import numpy as np
import logging as log
from leaver_game import BaseLine, CommNet


N_AGENTS = 50
VECTOR_LEN = 128
BATCH_SIZE = 64
LEVER = 5


# set logger
log.basicConfig(level=log.INFO, filename="leaver_train.log")
console = log.StreamHandler()
console.setLevel(log.INFO)
log.getLogger("").addHandler(console)


def train(episode):
    actor = CommNet(num_agents=N_AGENTS, vector_len=VECTOR_LEN, batch_size=BATCH_SIZE)
    critic = BaseLine(num_agents=N_AGENTS, vector_len=VECTOR_LEN, batch_size=BATCH_SIZE)

    for i in range(episode):
        ids = np.array([np.random.choice(N_AGENTS, LEVER, replace=False)
                        for _ in range(BATCH_SIZE)])

        reward = actor.get_reward(ids)
        baseline = critic.get_reward(ids)

        actor.train(ids, base_line=baseline, base_reward=reward, itr=i, log=log)
        critic.train(ids, base_reward=reward, itr=i, log=log)


if __name__ == "__main__":
    train(100000)

