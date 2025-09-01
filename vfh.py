import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.signal import savgol_filter
from typing import Tuple


def _angle_difference(angle1, angle2) -> float:
    """
    :param angle1: radian
    :param angle2: radian
    :return: 두 각의 차이
    """
    tmp = abs(angle1 - angle2) % (2 * np.pi)
    return min(tmp, 2 * np.pi - tmp)


def _middle_section(a, b, bins) -> int:
    mid = (a + b) // 2 % bins
    cand = (a + b + bins) // 2 % bins
    if min(abs(a - mid), abs(b - mid)) > min(abs(a - cand), abs(b - cand)):
        mid = cand
    return mid


def vfh(grid: np.ndarray, target_angle: float,
        threshold: int = 1, smax: int = 80 // 4,
        bins: int = 80) -> Tuple[np.ndarray, float, float, float, bool]:
    theta_interval = (2 * np.pi) / bins
    bin_centers = []
    for i in range(bins):
        bin_centers.append(-np.pi + (2 * np.pi) * i / bins + theta_interval / 2)
    origin = (grid.shape[0] // 2, grid.shape[1] // 2)

    obstacle_pixels = np.argwhere(grid > 0)
    if obstacle_pixels.size == 0:
        return np.zeros((bins,)), 0, bin_centers[((-smax + bins) // 2) % bins], bin_centers[
            ((smax + bins) // 2) % bins], True
    rel_obs = obstacle_pixels - origin
    dists = np.linalg.norm(rel_obs, axis=1)

    angles = np.arctan2(rel_obs[:, 1], rel_obs[:, 0])

    danger = np.zeros((bins,))
    dist_max = np.sqrt(grid.shape[0] ** 2 + grid.shape[1] ** 2)
    for dist, angle in zip(dists, angles):
        q = (angle + np.pi) / theta_interval
        # danger[max(0, min(bins - 1, round(q)))] += 1 - dist / dist_max
        danger[max(0, min(bins - 1, round(q)))] += 100 / (dist ** (3 / 2))
    danger = savgol_filter(danger, 5, 2)

    under_thre = danger < threshold

    is_safe = np.zeros_like(under_thre)

    i = 0
    all_safe = False
    while i < danger.shape[0]:
        if under_thre[i]:
            j = i
            while under_thre[j]:
                j = (j + 1) % (danger.shape[0])
                if j == i:
                    all_safe = True
                    break
            if all_safe:
                break
            if (j - i) % danger.shape[0] > smax:
                k = i
                while k != j:
                    is_safe[k] = 1
                    k = (k + 1) % (danger.shape[0])

            if j < i:
                break
            i = j
        else:
            i += 1
    if all_safe:
        return danger, 0, bin_centers[((-smax + bins) // 2) % bins], bin_centers[((smax + bins) // 2) % bins], True

    kn = bins // 2  # TODO
    min_angle_diff = float('inf')
    for i in range(bins):
        if not is_safe[i]:
            continue
        cur_angle_diff = _angle_difference(bin_centers[i], target_angle)
        if min_angle_diff > cur_angle_diff:
            kn = i
            min_angle_diff = cur_angle_diff

    kf = (kn + smax) % bins
    if not is_safe[_middle_section(kn, kf, bins)]:
        kf = (kn - smax) % bins
    else:
        if _angle_difference(bin_centers[(kn - smax) % bins], target_angle) > _angle_difference(bin_centers[kf],
                                                                                                target_angle):
            kf = (kn - smax) % bins
        if not is_safe[_middle_section(kn, kf, bins)]:
            kf = (kn + smax) % bins

    kn_angle = bin_centers[kn]
    kf_angle = bin_centers[kf]
    mid_section = _middle_section(kn, kf, bins)
    is_final_angle_section_safe = is_safe[mid_section]
    final_angle = bin_centers[mid_section]
    # final_angle = np.arctan2(np.sin(kn_angle)+np.sin(kf_angle), np.cos(kn_angle)+np.cos(kf_angle))

    return danger, final_angle, kn_angle, kf_angle, is_final_angle_section_safe
