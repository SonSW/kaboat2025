import serial
import time

ser = serial.Serial('/dev/ttyUSB0', 9600)

while True:
    ser.write(b'a')
    time.sleep(1.5)
    ser.write(b'b')
    time.sleep(1.5)
