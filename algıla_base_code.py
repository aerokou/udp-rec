import cv2
import torch
import os

# YOLOv5 modelini yükle
model = torch.hub.load('./yolov5', 'custom', f'{os.getcwd()}/best.pt', source='local')
# img = ["C:\\Users\\egepa\\OneDrive\\Resimler\\Film Rulosu\\WIN_20240426_18_40_59_Pro.jpg"]

# Kamera bağlantısını kur
cap = cv2.VideoCapture("./WIN_20240426_18_41_04_Pro (2).jpg")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Algılama başladı")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
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

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
