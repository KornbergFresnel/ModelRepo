import gym
import random
from ..utils import imresize


class Env(object):
    def __init__(self, config):
        """You can import environment configuration to create a new gym-like environment.
        """
        self.env = gym.make(config)

        screen_height, screen_width, self.action_repeat, self.random_start = \
                config.obs_width,  config.obs_height, config.action_repeat, config.random_start
        self.display = config.display
        sele.display_area = (screen_width, screen_height)
        
        self._obs = None
        self.reward = 0.0
        self.terminal = True

    @property
    def obs(self):
        """Preprocessing the raw observation
        """

        # TODO: convert raw observation to input observation
        return

    @property
    def state(self):
        return self._obs, self.reward, self.terminal

    @property
    def lives(self):
        return self.env.ale.lives()

    @property
    def action_size(self):
        return self.env.action_space.n

    def new_game(self, from_random_game=False):
        """Create a new game
        """

        if self.alivs == 0:
            self.env.reset()
        self._step(0)
        self._render()
        return self.obs, 0, 0, self.terminal

    def new_random_game(self):
        """Make a new random game, then return the state after several
        steps.
        """

        self.new_game(from_random_game=True)

        for _ in range(random.choice(0, self.random_start - 1)):
            self._step(0)
        self.render()
        
        return self.obs, 0, 0, self.terminal

    def _step(self, action):
        """Accept a action, then update game state: `(self._obs, self.reward, self.terminal, _)`
        if action == 0, states game recover
        """

        self._obs, self.reward, self.terminal, _ = self.env.step(action)

    def _random_step(self, action):
        """Random sample an action to execute.
        """

        action = self.env.action_space.sample()
        self._step(action)

    def render(self):
        if self.display:
            self.env.render()

    def after_act(self, action):
        """After the agent do a action, the environment will render again."""

        self.render()


class GymEnvironment(Env):
    def __init__(self, config):
        super().__init__(config)

    def act(self, action, is_training=True):
        """According to the paper which proposed by DeepMind, before an action make sense,
        we would better do it `self.action_repeat` """
        sum_reward = 0.0
        s_lives = self.lives  # get the livs at begining
        
        # action repeat
        for _ in range(self.action_repeat):
            self._step(action)  # each step will update the obs, reward and terminal value
            sum_reward += self.reward

            if is_training and s_lives > self.lives:
                sum_reward -= 1
                self.terminal = True

            if self.terminal:
                break

        self.reward = sum_reward

        self.after_act(action)
        return self.state


class SimpleGymEnvironment(Env):
    def __init__(self, config):
        super().__init__(config)

    def act(self, action, is_training=True):
        self._step(action)
        self.after_act(action)
        return self.state




    
