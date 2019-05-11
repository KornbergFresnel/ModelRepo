def linear_decay(data, decay_factor, min_val, maxval):
    return max(min_val, min(data - decay_factor, maxval))


def exponential_decay(data, decay_factor):
    raise NotImplementedError

