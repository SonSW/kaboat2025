import numpy as np
import ydlidar
from preprocessing import lidar_transform


class Lidar:

    def __init__(self, port, baud_rate, rmax=16.0, do_filter=False):
        """
        :param port: /dev/ttyXXX
        :param baud_rate:
        :param rmax: LiDAR 최대 측정 거리 (미터)
        :param do_filter: 필터링을 할 것인지. (아직 구현 안 됨)
        """
        self.rmax = rmax
        self.do_filter = do_filter
        if self.do_filter:
            raise NotImplementedError("아직 필터 기능 없음.")  # 굳이 있어야 될까?

        self.laser = ydlidar.CYdLidar()
        self.laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
        self.laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, baud_rate)

        self.scan = ydlidar.LaserScan()

        self.laser.initialize()
        self.laser.turnOn()

    def get_coords(self) -> np.ndarray:
        """
        라이다로 인식된 모든 점의 좌표를 (x좌표, y좌표) 형태로 구한다.
        :return: Nx2 array
        """
        self.laser.doProcessSimple(self.scan)
        points = self.scan.points

        ranges = np.fromiter((p.range for p in points), np.float32)
        angles = np.fromiter((p.angle for p in points), np.float32)

        coords = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
        coords = coords[ranges <= self.rmax]

        if self.do_filter:
            raise NotImplementedError()
        else:
            return coords

    def get_grid(self, map_size: int, pitch: float, roll: float):
        """
        한 변의 길이가 map_size인 맵을 구한다.
        맵의 각 칸은 0 또는 1. (1이 장애물)

        :param map_size: 정사각형 grid의 한 변의 길이
        :param pitch: pitch (radian)
        :param roll: roll (radian)
        :return: map_size x map_size array
        """
        scaled_coords = lidar_transform(self.get_coords(), 0, pitch, roll)
        scaled_coords[:, 0] *= map_size / (2 * self.rmax)
        scaled_coords[:, 1] *= map_size / (2 * self.rmax)
        scaled_coords += np.array((map_size / 2, map_size / 2))
        int_coords = np.round(scaled_coords).astype(int)
        int_coords[:, 0] = np.clip(int_coords[:, 0], 0, map_size - 1)
        int_coords[:, 1] = np.clip(int_coords[:, 1], 0, map_size - 1)
        map_grid = np.zeros((map_size, map_size))
        map_grid[int_coords[:, 0], int_coords[:, 1]] = 1
        map_grid[map_size // 2, map_size // 2] = 0
        return map_grid.copy()

    def disconnect(self):
        self.laser.turnOff()
        self.laser.disconnecting()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def __del__(self):
        self.disconnect()
