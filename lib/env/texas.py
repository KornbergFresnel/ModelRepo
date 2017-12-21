import gym
import numpy as np
import sys
import io
from gym.envs.toy_text import discrete


class TexasEnv(discrete.DiscreteEnv):

    metadata = {"renders.modes": ["human", "ansi"]}

    def __init__(self):
        # === define environment ===
        self.shape = None
        nS, nA, P, isd = None, None, None, None
        super(TexasEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return

        out_file = io.StringIO() if mode == "ansi" else sys.stdout

        for s in range(self.nS):
            # TODO: implement rule
            output = None
            out_file.write(output)

        out_file.write("\n")


