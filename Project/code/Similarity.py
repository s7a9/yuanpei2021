import numpy as np


def calc_cos_similarity(vec1, vec2):
    vec1 = np.mat(vec1)
    vec2 = np.mat(vec2)
    print((vec1 * vec2.T).shape)
    num = float(vec1 * vec2.T)
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return 0.5 + 0.5 * num / denom
