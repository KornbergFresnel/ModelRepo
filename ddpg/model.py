import numpy as np
import tensorflow as tf

from lib.common import Buffer, BaseModel
from lib.tools import flatten


class DDPG(BaseModel):
    def __init__(self, sess, ):