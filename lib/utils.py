import numpy as np


class History(object):
    def __init__(self, config):
        """History implements the frames stacking, its inner data-structure
        is related with data format (`NCHW` or `NHWC`)
        """

        self.cnn_format = config.cnn_format
        obs_height, obs_width, history_length = \
                config.obs_height, config.obs_width, config.history_length

        if self.cnn_format == "NCHW":
            self._history = np.zeros(shape=(history_length, obs_height, obs_width), dtype=np.float32)
        else:
            self._history = np.zeros(shape=(obs_height, obs_width, history_length), dtype=np.float32)

    def add(self, obs_frame):
        """Accept an screen frame then append it to the tail while pop the head frame
        """
        # replace the former `history_length - 1` with the last `history_length - 1` frames
        self._history[:-1] = self._history[1:]
        self._history[-1] = obs_frame

    def get(self):
        return self._history



