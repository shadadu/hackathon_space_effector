import numpy as np

def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quat_inverse(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w]) / np.dot(q, q)

def small_angle_quat(dtheta):
    return quat_normalize(np.array([
        0.5 * dtheta[0],
        0.5 * dtheta[1],
        0.5 * dtheta[2],
        1.0
    ]))
