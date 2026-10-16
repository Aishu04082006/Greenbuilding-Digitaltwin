"""Computer vision: people (YOLOv8), lights (brightness), fan (blade motion), AC (indicator LED)."""
import cv2

_model = None
_prev = {}


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")      # downloads automatically the first time
    return _model


def _box(frame, roi):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi                 # fractions 0..1
    return int(x1 * w), int(y1 * h), max(int(x2 * w), int(x1 * w) + 2), max(int(y2 * h), int(y1 * h) + 2)


def fan_spinning(frame, roi, thr=5.0, key="fan"):
    """Spinning blades change the pixels in the fan area between frames."""
    x1, y1, x2, y2 = _box(frame, roi)
    g = cv2.GaussianBlur(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (5, 5), 0)
    prev = _prev.get(key)
    _prev[key] = g
    if prev is None or prev.shape != g.shape:
        return False, 0.0
    m = float(cv2.absdiff(g, prev).mean())
    return m > thr, m


def ac_indicator(frame, roi, thr=0.0008):
    """AC units show a small bright coloured LED/display when running."""
    x1, y1, x2, y2 = _box(frame, roi)
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    ratio = float(((hsv[..., 2] > 220) & (hsv[..., 1] > 120)).mean())
    return ratio > thr, ratio


def analyze(frame, light_threshold=90, fan_roi=(0.3, 0.0, 0.7, 0.3),
            ac_roi=(0.7, 0.0, 1.0, 0.25), conf=0.4, key="cam"):
    result = _get_model()(frame, verbose=False, conf=conf, classes=[0])[0]
    people = len(result.boxes)
    fh, fw = frame.shape[:2]
    # where each person stands: (x centre, feet y) as fractions of the frame -> drives the 3D figures
    centers = [(float((b[0] + b[2]) / 2 / fw), float(b[3] / fh))
               for b in result.boxes.xyxy.cpu().numpy()]
    brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    lights = brightness > light_threshold
    fan, fan_m = fan_spinning(frame, fan_roi, key=key)
    ac, ac_r = ac_indicator(frame, ac_roi)

    img = result.plot()
    for roi, name, on, col in ((fan_roi, "FAN", fan, (255, 200, 0)), (ac_roi, "AC", ac, (0, 200, 255))):
        x1, y1, x2, y2 = _box(img, roi)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"{name}: {'ON' if on else 'off'}", (x1 + 4, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.putText(img, f"People: {people}  Lights: {'ON' if lights else 'off'}", (10, img.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return dict(people=people, lights=lights, brightness=brightness, fan=fan, ac=ac,
                centers=centers, fan_motion=fan_m, ac_ratio=ac_r, annotated=img)
