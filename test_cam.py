import cv2
from vision import analyze

cap = cv2.VideoCapture(0)   # try 1 if 0 fails
while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera not opened"); break
    people, lights, b, img = analyze(frame)
    cv2.putText(img, f"People: {people}  Lights: {lights}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Test", img)
    if cv2.waitKey(1) == 27:   # Esc to quit
        break
cap.release()
cv2.destroyAllWindows()