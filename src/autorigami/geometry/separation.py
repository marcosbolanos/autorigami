from scipy.spatial import cKDTree
import numpy as np


def validate_non_self_intersection(
    min_euclid_distance: float,
    n_ignored_adjacent_edges: int = 1
):

    return

def

def segment_segment_distance(p0, p1, q0, q1):
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)

    denom = a * c - b * b

    if denom > 1e-12:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
    else:
        # segments are almost parallel, don't divide by zero
        s = 0.0
        t = e / c if c > 1e-12 else 0.0

    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    # recompute after clipping one variable
    # because clipping s or t can change the best value of the other
    if c > 1e-12:
        t = np.clip((np.dot(v, p0 + s*u - q0)) / c, 0.0, 1.0)
    if a > 1e-12:
        s = np.clip((-np.dot(u, q0 + t*v - p0)) / a, 0.0, 1.0)

    closest_p = p0 + s * u
    closest_q = q0 + t * v

    return np.linalg.norm(closest_p - closest_q), closest_p, closest_q
