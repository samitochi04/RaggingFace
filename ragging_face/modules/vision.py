try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

# delay YOLO import until needed
_model = None


def _import_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        return None

def load_model():
    global _model
    if _model is None:
        if cv2 is None:
            raise RuntimeError("OpenCV not available; cannot load vision model")
        YoloCls = _import_yolo()
        if YoloCls is None:
            raise RuntimeError("YOLO (ultralytics) library not installed")
        _model = YoloCls('yolov8n.pt')
    return _model


def detect_defects(image_path: str):
    """Run YOLO detection on the provided image and return results.

    Args:
        image_path: path to the image file.

    Returns:
        dict with keys: 'image', 'boxes', 'scores', 'labels'
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is not available; vision module cannot run")
    model = load_model()
    results = model(image_path)
    # results is a list of Results objects; take first
    r = results[0]
    boxes = []
    # draw boxes on image copy
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    names = model.model.names if hasattr(model, 'model') and hasattr(model.model, 'names') else {}
    for box, score, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
        x1, y1, x2, y2 = map(int, box.tolist())
        label = int(cls)
        label_name = names.get(label, str(label))
        boxes.append({
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'score': float(score),
            'label': label_name
        })
        color = (255, 0, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label_name}:{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return {'image': image, 'boxes': boxes}
