import os, sys
import argparse

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from model import MultiAgent
from tools import ReplayBuffer
from config import GeneralConfig


REPLAY_BUFFER = []


def play(n_round, _env, _policies, train=False):
    obs_n = _env.reset()
    terminate = False

    step_counter = 0
    print("\n--- ROUND #{} ---\n".format(n_round))
    while not terminate and step_counter < 1000:
        act_n = _policies.act(obs_n)

        obs_n_next, reward_n, done_n, _ = _env.step(act_n)
        REPLAY_BUFFER.push(obs_n, act_n, reward_n, done_n)
        obs_n = obs_n_next

        terminate = any(done_n)
        step_counter += 1

        if step_counter % 100 == 0:
            print("> step: {0}, reward: {1}".format(step_counter, np.around(reward_n, decimals=6)))

    if len(REPLAY_BUFFER) > REPLAY_BUFFER.batch_size:
        loss, eval_q, time_com = _policies.train(REPLAY_BUFFER)
        print("\n[* TRAIN] Loss{0}, eval-Q: {1}, time-Com: {2:.3f}".format(loss, eval_q, time_com))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scenario', default="simple_speaker_listener.py",
                        help="Path of the scenario Python script.")
    parser.add_argument('-n', '--n_round', type=int, default=600, help="Number of round you wanna run.")
    parser.add_argument('-d', '--dir', type=str, default="./models", help="Grandparent directory path to store model.")
    parser.add_argument('-e', '--every', type=int, default=10, help="Save model at each x steps")
    parser.add_argument('-l', '--load', type=int, help="Indicates the step you wanna start, file must exist")

    args = parser.parse_args()
    start = 0

    scenario = scenarios.load(args.scenario).Scenario()
    world = scenario.make_world()

    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    policies = MultiAgent(env, "Multi-Agent")

    REPLAY_BUFFER = ReplayBuffer(env, GeneralConfig())

    if args.load is not None:
        start = args.load

        if args.dir is None:
            print("[!] Please indicate a path for model storing")
            exit(1)

        policies.load(args.dir, start)

    for i in range(start, start + args.n_round):
        play(i, env, policies, train=True)

        if (i + 1) % args.every == 0:
            policies.save(args.dir, i)
