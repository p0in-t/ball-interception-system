import numpy as np

def euclidean_norm(v, w):
    dist = np.sqrt(np.sum(np.square(v-w)))
    return dist