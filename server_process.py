import io
import socket
import struct
from PIL import Image
import cv2
import numpy
import sys
import torch
import os

model = torch.hub.load('./yolov5', 'custom', f'{os.getcwd()}/best.pt', source='local')
server_socket = socket.socket()
server_socket.bind((sys.argv[1], int(sys.argv[2])))  
server_socket.listen(0)
print("Listening")
connection = server_socket.accept()[0].makefile('rb')

try:
    img = None
    while True:
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        image_stream.seek(0)
        image = Image.open(image_stream)
        frame = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR) # goruntu isleme yapmak icin hazir hale getiriyoruz
        # Görüntüyü tensöre formatına dönüştür (BGR -> RGB)
        frame_rgb = frame[:, :, ::-1]

        # YOLOv5 modeli ile nesne algılama yap
        results = model(frame_rgb)

        # Algılama sonuçlarını al
        pred = results.pred[0]

        # Algılama sonuçlarını çerçevelerle görselleştir
        for obj in pred:
            class_id = int(obj[-1])
            score = float(obj[-2])
            # label = model.module.names[class_id]
            label = "fire-poster"
            if score > 0.7:  # Belirli bir güven eşiği üzerindeki sonuçları göster
                x1, y1, x2, y2 = map(int, obj[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Sonuçları göster
        cv2.imshow('YOLOv5 Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
finally:
    cv2.destroyAllWindows()
    connection.close()
    server_socket.close()
