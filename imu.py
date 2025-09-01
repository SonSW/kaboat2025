import sys
import numpy as np
from globals import MSCL_PACKAGE_PATH

sys.path.append(MSCL_PACKAGE_PATH)
import mscl


class IMU:

    def __init__(self, port: str, baud_rate: int, virtual: bool = False):
        self.virtual = virtual

        if not virtual:
            self.connection = mscl.Connection.Serial(port, baud_rate)
            self.node = mscl.InertialNode(self.connection)
            self.node.setToIdle()

            ahrs_channels = mscl.MipChannels()
            channel_ids = [
                # mscl.MipTypes.CH_FIELD_SENSOR_RAW_ACCEL_VEC,
                # mscl.MipTypes.CH_FIELD_SENSOR_RAW_GYRO_VEC,
                # mscl.MipTypes.CH_FIELD_SENSOR_RAW_MAG_VEC,
                mscl.MipTypes.CH_FIELD_SENSOR_SCALED_ACCEL_VEC,
                mscl.MipTypes.CH_FIELD_SENSOR_SCALED_GYRO_VEC,
                mscl.MipTypes.CH_FIELD_SENSOR_SCALED_MAG_VEC,
                # mscl.MipTypes.CH_FIELD_SENSOR_STABILIZED_MAG_VEC,
                # mscl.MipTypes.CH_FIELD_SENSOR_STABILIZED_ACCEL_VEC,
                mscl.MipTypes.CH_FIELD_SENSOR_EULER_ANGLES,
            ]
            for channel_id in channel_ids:
                ahrs_channels.append(mscl.MipChannel(
                    channel_id, mscl.SampleRate.Hertz(100)
                ))

            # set the active channels for the AHRS/IMU class on the Node
            self.node.setActiveChannelFields(mscl.MipTypes.CLASS_AHRS_IMU, ahrs_channels)

            self.node.enableDataStream(mscl.MipTypes.CLASS_AHRS_IMU)
            self.node.resume()

    def get_imu_data_dict(self, timeout_ms=500):
        if self.virtual:
            return {}
        else:
            try:
                data_dict = {}

                packets = self.node.getDataPackets(timeout_ms)
                for packet in packets:
                    for dataPoint in packet.data():
                        data_dict[dataPoint.channelName()] = dataPoint.as_float()
                # print("data_dict:")
                # print(data_dict)
                return data_dict

            except:
                return None

    def get_roll_pitch_yaw(self, timeout_ms=500):
        di = self.get_imu_data_dict(timeout_ms)
        if di is None:
            return None, None, None
        else:
            return di['roll'], di['pitch'], di['yaw']

    def get_compass_angle(self, timeout_ms=500):
        di = self.get_imu_data_dict(timeout_ms)
        if di is None:
            return None
        else:
            vec = np.array([di["scaledMagX"], di["scaledMagY"]], dtype=float)
            x = vec[0]
            y = vec[1]

            # 임시방편임.
            vec[1] = -x
            vec[0] = y

            return np.arctan2(vec[1], vec[0])
