import socket
import struct
import time
from picamera2 import Picamera2
import cv2
import sys

print(f"Server bekleniyor {sys.argv[1]}:{sys.argv[2]}")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((sys.argv[1], int(sys.argv[2])))

connection = client_socket.makefile('wb')

try:
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)}))
    picam2.start()
    time.sleep(2)  # Kamera başlatma süresi

    while True:
        frame = picam2.capture_array()
        _, buffer = cv2.imencode('.jpg', frame)
        connection.write(struct.pack('<L', len(buffer)))
        connection.write(buffer)

finally:
    connection.write(struct.pack('<L', 0))
    connection.close()
    client_socket.close()
