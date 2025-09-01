from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
import time
import random


def _angle_of_data(data: np.ndarray) -> float:
    data_centered = data - np.mean(data, axis=0)
    cov = np.cov(data_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.abs(eigvals).argmax()]
    return np.arctan2(v[1], v[0])


def _mse(P, R, t, Q, kd_tree_Q=None):
    if kd_tree_Q is None:
        kd_tree_Q = KDTree(Q)
    transformed_P = (R @ P.T).T + t
    mse = np.linalg.norm(transformed_P - Q[kd_tree_Q.query(transformed_P, workers=-1)[1]], axis=1).sum() / len(P)
    return mse


def _ICP_2d(P: np.ndarray, Q: np.ndarray, theta_init, num_iter=1000, eps=1e-4):
    kd_tree = KDTree(Q)

    p_bar = P.mean(axis=0)
    P_resid = P - p_bar

    R = np.array([[np.cos(theta_init), -np.sin(theta_init)], [np.sin(theta_init), np.cos(theta_init)]])
    t = Q.mean(axis=0) - p_bar

    rmse_prev = float('inf')
    for k in range(num_iter):
        _, Q_hat_idx = kd_tree.query((R @ P.T).T + t, workers=-1)
        Q_hat = Q[Q_hat_idx]
        q_hat_bar = Q_hat.mean(axis=0)
        S = P_resid.T @ (Q_hat - q_hat_bar)
        U, Sigma, VT = np.linalg.svd(S)
        R = VT.T @ np.diag([1, np.linalg.det(VT.T @ U.T)]) @ U.T
        t = q_hat_bar - (R @ p_bar)

        mse = _mse(P, R, t, Q, kd_tree_Q=kd_tree)
        rmse = np.sqrt(mse)
        # print(f"rmse={rmse}, rmse_prev={rmse_prev}")
        if abs(rmse - rmse_prev) < eps:
            break
        rmse_prev = rmse

    return R, t


def ICP_2d_with_tries(P: np.ndarray, Q: np.ndarray, theta_cands, num_iter=1000, eps=1e-4):
    kd_tree = KDTree(Q)
    p_bar = P.mean(axis=0)

    final_R = np.eye(2)
    final_t = Q.mean(axis=0) - p_bar
    best_mse = float('inf')

    for theta in theta_cands:
        R, t = _ICP_2d(P, Q, theta, num_iter, eps)

        mse = _mse(P, R, t, Q, kd_tree_Q=kd_tree)
        if mse < best_mse:
            final_R = R.copy()
            final_t = t.copy()
            best_mse = mse

    return final_R, final_t


def ICP_2d(P: np.ndarray, Q: np.ndarray, theta_init: float = None, num_iter=1000, eps=1e-4):
    if theta_init is None:
        theta = abs(_angle_of_data(Q) - _angle_of_data(P))
        print(f"theta init: {theta * 180 / np.pi} deg")
        return ICP_2d_with_tries(P, Q, [theta, np.pi - theta], num_iter, eps)
    else:
        return ICP_2d_with_tries(P, Q, [theta_init], num_iter, eps)


def ICP_2d_with_RANSAC(P: np.ndarray, Q: np.ndarray, theta_init: float = None, num_iter_ransac=5, num_iter=1000,
                       eps=1e-4):
    p_bar = P.mean(axis=0)

    final_R = np.eye(2)
    final_t = Q.mean(axis=0) - p_bar
    best_mse = float('inf')

    for _ in range(num_iter_ransac):
        P_sampled = P[np.random.randint(len(P), size=int(len(P) * 0.3)), :]
        R, t = ICP_2d(P_sampled, Q, theta_init, num_iter=num_iter, eps=eps)
        mse = _mse(P_sampled, R, t, Q)
        if mse < best_mse:
            best_mse = mse
            final_R = R.copy()
            final_t = t.copy()

    return final_R, final_t


# N = 400
# P = np.random.randn(2 * N).reshape((N, 2))

# P = []
# for i in range(200):
#     P.append((i*1/200, 1))
# for i in range(100):
#     P.append((2, 0.5/100*i))
# P += (0.07*np.random.randn(2 * N)).reshape((N, 2)).tolist()
# P += [(random.random(), random.random()*2) for _ in range(30)]
# P = np.array(P)
# P += (0.02*np.random.randn(2 * len(P))).reshape((len(P), 2))
#
# theta = np.pi / 7
# # assert abs(theta) < np.pi / 3
# Q = (np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ P.T).T + np.array([0.11, -0.01])
# # Q += np.random.randn(2 * N).reshape((N, 2)) * 0.05
#
# P = P[np.random.randint(len(P), size=int(len(P) * 0.8)), :]
# Q = Q[np.random.randint(len(Q), size=int(len(Q) * 0.8)), :]
#
# theta_init_list = np.linspace(-np.pi/6, np.pi/6, 5)
#
# start_time = time.time()
# R, t = ICP_2d_with_tries(P, Q, theta_init_list)
# end_time = time.time()
# print("Calculation time:", end_time - start_time, "sec")
#
# print("Rotation Matrix:")
# print(R)
# print("Rotation Angle:")
# print(np.arccos(R[0, 0]) * 180.0 / np.pi, "degree")
# print("Translation Vector:")
# print(t)
# estimated_Q = (R @ P.T).T + t
#
# plt.scatter(Q[:, 0], Q[:, 1], c='r', alpha=0.5)
# plt.scatter(estimated_Q[:, 0], estimated_Q[:, 1], c='b', alpha=0.5)
# # plt.scatter(P[:, 0], P[:, 1], c='b', marker='x', alpha=0.2)
# plt.show()
