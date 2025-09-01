import numpy as np


def lidar_transform(coords: np.ndarray, yaw, pitch, roll) -> np.ndarray:
    is_single_vec = coords.ndim == 1
    if is_single_vec:
        coords = coords.reshape(1, -1)

    yaw_mat = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    result = (yaw_mat @ coords.T).T
    cos_tilt_angle = abs(np.cos(pitch) * np.cos(roll))
    result[:, 0] *= cos_tilt_angle

    if is_single_vec:
        return result[0]
    else:
        return result


def convert_lidar_coords_to_grid(coords: np.ndarray, map_size, lidar_rmax, do_clip=True) -> np.ndarray:
    scaled_coords = coords.copy()
    scaled_coords[:, 0] *= map_size / (2 * lidar_rmax)
    scaled_coords[:, 1] *= map_size / (2 * lidar_rmax)
    scaled_coords += np.array((map_size / 2, map_size / 2))
    int_coords = np.round(scaled_coords).astype(int)
    if do_clip:
        int_coords[:, 0] = np.clip(int_coords[:, 0], 0, map_size - 1)
        int_coords[:, 1] = np.clip(int_coords[:, 1], 0, map_size - 1)
    map_grid = np.zeros((map_size, map_size))
    map_grid[int_coords[:, 0], int_coords[:, 1]] = 1
    map_grid[map_size // 2, map_size // 2] = 0
    return map_grid.copy()


def convert_coord_to_pixel(coord, map_size, lidar_rmax, do_clip=True) -> np.ndarray:
    assert coord.ndim == 1
    scaled_coord = coord.copy()
    scaled_coord[0] *= map_size / (2 * lidar_rmax)
    scaled_coord[1] *= map_size / (2 * lidar_rmax)
    scaled_coord += np.array((map_size / 2, map_size / 2))
    int_coord = np.round(scaled_coord).astype(int)
    if do_clip:
        int_coord[0] = np.clip(int_coord[0], 0, map_size - 1)
        int_coord[1] = np.clip(int_coord[1], 0, map_size - 1)
    return int_coord


def grid_to_coords(map_grid: np.ndarray, size, lidar_max) -> np.ndarray:
    return (np.argwhere(map_grid == 1) - np.array((size / 2, size / 2))) * (2 * lidar_max / size)
