import socket
import threading
import time
import traceback

import numpy as np
import serial
from pynmeagps import SocketWrapper
from pyubx2 import (
    UBXReader, UBX_PROTOCOL, NMEA_PROTOCOL, RTCM3_PROTOCOL
)
from typing import Tuple

# 지구 반지름
_R = 6.3781 * (10 ** 6)


class GPSRTK:
    """
    실시간으로 RTK 서버와 통신하여 현재 위도와 경도를 구한다.
    """
    def __init__(self, port: str, baud_rate: int, init_lat=None, init_lon=None):
        """
        :param port: /dev/ttyXXX 형태
        :param baud_rate: GPS <-> 컴퓨터 통신할 때 쓸 baud rate
        :param init_lat: 초기 위도. 기본값 None
        :param init_lon: 초기 경도. 기본값 None
        """
        # GPSRTK 객체를 만들 때 실행됨

        self.init_lat = init_lat
        self.init_lon = init_lon

        # GPS와 직접적으로 통신하는 시리얼통신
        self.ser = serial.Serial(port, baud_rate)

        # Thread locks
        self.lock_gga = threading.Lock()
        self.lock_latlon = threading.Lock()

        self.last_gga_decoded = None

        self.lat = 0 if init_lat is None else init_lat
        self.lon = 0 if init_lon is None else init_lon
        self.last_measure_time = 0

        # RTK 서버와의 통신을 준비
        self.socket = None
        self._prepare_socket()

        # 본격적으로 실시간 업데이트를 시작
        threading.Thread(target=self._pump_rtcm_to_receiver).start()
        threading.Thread(target=self._update_lat_lon).start()

    def get_lat_lon(self) -> Tuple[float, float, float]:
        """
        현재(정확히는 가장 최근에 업데이트된 값인) 위도와 경도, 그리고 그것을 구했을 때의 시간(Unix time)을 리턴한다.

        :return: lat, lon, last_measure_time; 각각 위도, 경도, 그리고 그것을 측정했을 때의 시간(Unix time).
        """
        with self.lock_latlon:
            lat = self.lat
            lon = self.lon
            last_measure_time = self.last_measure_time

        # 만약 아직 한 번도 GPS로 위도경도를 구한 적이 없다면,
        if last_measure_time == 0:
            last_measure_time = time.time()
            self.last_measure_time = last_measure_time
            # 이 경우, 함수의 리턴값은 (0, 0, 현재시간)이 된다.
            # TODO: 이것이 문제를 일으킬수도?

        return lat, lon, last_measure_time

    def get_xy_coord(self, lat=None, lon=None) -> Optional[np.ndarray]:
        # TODO
        """초기 위도/경도를 가지고, x좌표와 y좌표를 구한다."""
        with self.lock_latlon:
            init_lat, init_lon = self.init_lat, self.init_lon

        if init_lat is None or init_lon is None:
            return None  # TODO: 이러면 안 좋을듯..
        if lat is None or lon is None:
            lat, lon, _ = self.get_lat_lon()

        lat_diff_rad = (lat - init_lat) * np.pi / 180
        lon_diff_rad = (lon - init_lon) * np.pi / 180

        x = _R * lon_diff_rad * np.cos(lat_diff_rad)
        y = _R * lat_diff_rad

        return np.array([x, y])

    def _prepare_socket(self):
        while True:
            try:
                server = "www.gnssdata.or.kr"
                port = 2101
                request_headers = ("GET /GUMC-RTCM31 HTTP/1.1\r\n"
                                   "Host: www.gnssdata.or.kr:2101\r\n"
                                   "User-Agent: NTRIP pygnssutils/1.1.16\r\n"
                                   "Authorization: Basic c2V3b24xNDA3QGdtYWlsLmNvbTpnbnNz\r\n"
                                   "Ntrip-Version: Ntrip/2.0\r\n"
                                   "Accept: */*\r\n"
                                   # "Connection: close\r\n"
                                   "\r\n")
                self.sock = socket.create_connection((socket.gethostbyname(server), port))
                self.sock.sendall(request_headers.encode())
                hdr = self.sock.recv(1024)
                if b"ICY 200" not in hdr and b"200 OK" not in hdr:
                    print("Connecting to server failed. Retrying...")
                    time.sleep(2)
                    continue
            except:
                continue
            break

    def _pump_rtcm_to_receiver(self):
        # 나도 잘 모름

        def __gga_pusher():
            while True:
                with self.lock_gga:
                    _last_gga_decoded = self.last_gga_decoded
                if _last_gga_decoded:
                    try:
                        self.sock.sendall(
                            (_last_gga_decoded if _last_gga_decoded.endswith(
                                "\r\n") else _last_gga_decoded + "\r\n").encode())
                    except:
                        traceback.print_exc()
                        return
                time.sleep(5)

        threading.Thread(target=__gga_pusher, daemon=True).start()

        parser = UBXReader(
            SocketWrapper(self.sock, 1),
            bufsize=4096,
            labelmsm=True
        )
        while True:
            try:
                raw_data, parsed_data = parser.read()
                if not raw_data:
                    break
                self.ser.write(raw_data)
            except:
                traceback.print_exc()
                time.sleep(0.2)

    def _update_lat_lon(self):
        """self.lat과 self.lon을 실시간으로 업데이트."""

        ubr = UBXReader(
            self.ser,
            protfilter=NMEA_PROTOCOL | UBX_PROTOCOL | RTCM3_PROTOCOL,
            bufsize=4096,
        )
        while True:
            raw_data, parsed_data = ubr.read()
            identity = getattr(parsed_data, 'identity', "")
            if "GGA" in identity:
                with self.lock_gga:
                    self.last_gga_decoded = raw_data.decode().strip()
                qual = int(getattr(parsed_data, 'quality', '0'))
                # status = {0: "NO FIX", 1: "GPS", 2: "DGNSS", 4: "RTK FIX", 5: "RTK FLOAT", 6: "DR"}.get(qual, str(qual))

                # 만약 RTK FLOAT 또는 RTK FIX라면
                if qual in (4, 5):
                    with self.lock_latlon:
                        self.lat = parsed_data.lat
                        self.lon = parsed_data.lon
                        self.last_measure_time = time.time()
                        if self.init_lat is None or self.init_lon is None:
                            self.init_lat = self.lat
                            self.init_lon = self.lon
