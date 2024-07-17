import cv2

cam = cv2.VideoCapture(0)

while True:
    cap, res = cam.read()

    cv2.imshow("im",res)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break;

cam.release()
cv2.destroyAllWindows()