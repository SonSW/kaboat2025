import sys
import time
import traceback

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from globals import *
from gps import GPSRTK
from imu import IMU
from lidar import Lidar
from preprocessing import lidar_transform, convert_coord_to_pixel
from vfh import vfh

sys.path.append(MSCL_PACKAGE_PATH)
import mscl

MEASURE_PERIOD_SEC = 0.05
PATH_PLANNING_PERIOD_SEC = 0.1

LIDAR_RMAX = 2
MAP_SIZE = 200
MAP_SHAPE = np.array((MAP_SIZE, MAP_SIZE))
BINS = 80

GOAL_ABSCOORD = np.array([1, 0])

SHIP_ORIGIN_PIXEL = MAP_SHAPE // 2

# ==== VFH 히스토그램 시각화 세팅 (GPT가 해줌) ====
matplotlib.use('Agg')
plt.ion()
fig = Figure(figsize=(4, 3), dpi=100)
canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)
bars = ax.bar(range(BINS), [0] * BINS)
ax.set_ylim(0, 5)
ax.set_title("VFH Histogram")

cv2.namedWindow('Histogram', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Histogram', 400, 300)
# ===========================================

# LiDAR 맵 시각화 세팅
cv2.namedWindow('Live LIDAR Map', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Live LIDAR Map', MAP_SIZE * 3, MAP_SIZE * 3)

# 장치들 세팅
lidar = Lidar(LIDAR_PORT, LIDAR_BAUDRATE, rmax=LIDAR_RMAX)
imu = IMU(IMU_PORT, IMU_BAUDRATE, virtual=False)
gpsrtk = GPSRTK()  # TODO
arduino_serial = serial.Serial(ARDUINO_PORT, ARDUINO_BAUDRATE)

grid_map = np.zeros(MAP_SHAPE)


# =================================================================================

def _draw_square(img, x, y, size, color):
    """img의 x, y 위치에 한 변의 길이가 size이고 색깔이 color인 정사각형을 그림"""
    sh0, sh1, _ = img.shape

    for i in range(size):
        for j in range(size):
            curx = x + i - size // 2
            cury = y + j - size // 2
            if curx < 0 or curx >= sh0 or cury < 0 or cury >= sh1:
                continue
            img[curx, cury] = color


def in_my_way_mask(pixels: np.ndarray, safe_dist_pixels: float, goal: np.ndarray, origin: np.ndarray) -> np.ndarray:
    goal_vec = (goal - origin).reshape(1, -1)
    mat = np.eye(2) - (goal_vec.T @ goal_vec) / (np.linalg.norm(goal_vec) ** 2)
    mask = np.linalg.norm((mat @ (pixels - origin).T).T, axis=1) <= safe_dist_pixels
    mask &= np.dot(pixels - origin, goal - origin) > 0
    return mask


try:
    while True:
        time.sleep(MEASURE_PERIOD_SEC)

        # IMU 파트.
        # 1) Roll, Pitch, Yaw(compass_angle)을 구한다.
        # 2) 기욺각이 pi/15보다 크면, 그냥 현재 step 자체를 스킵한다.
        try:
            roll, pitch, _ = imu.get_roll_pitch_yaw()
            compass_angle = imu.get_compass_angle()
            if roll is None or pitch is None or compass_angle is None:
                continue
            if np.arccos(np.cos(roll) * np.cos(pitch)) > np.pi / 15:
                continue
        except mscl.Error_Connection:
            imu = IMU(IMU_PORT, IMU_BAUDRATE)
            continue
        except:
            traceback.print_exc()
            continue

        # Grid & Goal 파트.
        # 1) 화면에 표시할 grid map을 만든다.
        grid_map = lidar.get_grid(MAP_SIZE, pitch, roll)
        obs_pixels = np.argwhere(grid_map > 0)

        ship_origin_abscoord = gpsrtk.get_xy_coord()
        goal_pixel = convert_coord_to_pixel(
            lidar_transform(GOAL_ABSCOORD - ship_origin_abscoord, -compass_angle + np.pi / 2, pitch, roll),
            MAP_SIZE, lidar.rmax, do_clip=False)

        goal_vec = goal_pixel - SHIP_ORIGIN_PIXEL
        angle_to_goal = np.arctan2(goal_vec[1], goal_vec[0])
        danger, vfh_result_angle, kn_angle, kf_angle, is_safe = vfh(grid_map, angle_to_goal, bins=BINS)

        # Vis
        for i, b in enumerate(bars):
            b.set_height(danger[i])
        canvas.draw()
        buf = canvas.buffer_rgba()  # RGBA memoryview
        w, h = fig.canvas.get_width_height()
        hist_img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        hist_img_bgr = cv2.cvtColor(hist_img, cv2.COLOR_RGBA2BGR)
        cv2.imshow('Histogram', hist_img_bgr)

        img = cv2.cvtColor((grid_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Guidance
        is_goal_near = (0 <= goal_pixel[0] < MAP_SHAPE[0]) and (0 <= goal_pixel[1] < MAP_SHAPE[1])
        no_obstacles_in_way = np.all(
            ~in_my_way_mask(obs_pixels, safe_dist_pixels=10, goal=goal_pixel, origin=SHIP_ORIGIN_PIXEL))

        if is_goal_near and no_obstacles_in_way:  # 쭉 가면 됨
            forward_indicator_point = 10 * (goal_pixel - SHIP_ORIGIN_PIXEL) / np.linalg.norm(
                goal_pixel - SHIP_ORIGIN_PIXEL)
            forward_indicator_point += np.array([MAP_SIZE / 2, MAP_SIZE / 2])
        else:
            kn_point = 10 * np.array([np.cos(kn_angle), np.sin(kn_angle)])
            kn_point += np.array([MAP_SIZE / 2, MAP_SIZE / 2])

            kf_point = 10 * np.array([np.cos(kf_angle), np.sin(kf_angle)])
            kf_point += np.array([MAP_SIZE / 2, MAP_SIZE / 2])

            forward_indicator_point = 10 * np.array([np.cos(vfh_result_angle), np.sin(vfh_result_angle)])
            forward_indicator_point += np.array([MAP_SIZE / 2, MAP_SIZE / 2])

        # Vis
        _draw_square(img, MAP_SIZE // 2, MAP_SIZE // 2, 5, (0, 255, 0))
        if is_safe:
            _draw_square(img, round(forward_indicator_point[0]), round(forward_indicator_point[1]), 5, (0, 0, 255))
        if is_goal_near:
            _draw_square(img, goal_pixel[0], goal_pixel[1], 5, (255, 0, 0))

        # Arduino
        arduino_serial.write()

        cv2.imshow('Live LIDAR Map', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    lidar.disconnect()
