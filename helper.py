# some helper function for calculations in hmm model

import numpy as np

def logsumexp(a, axis=None):
    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    tmp = np.exp(a - a_max)
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis)
        out = np.log(s)

    return out+a_max




