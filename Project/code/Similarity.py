import numpy as np


def calc_cos_similarity(vec1, vec2):
    vec1 = np.mat(vec1)
    vec2 = np.mat(vec2)
    num = float(vec1 * vec2.T)
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return num / denom

def calc_euclidean_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dis = np.linalg.norm(vec1 - vec2)
    return 1 / (1 + dis)
