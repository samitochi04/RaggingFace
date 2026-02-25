# we'll use torchvision for object detection (no cv2 required)
try:
    import numpy as np
except ImportError:
    np = None

import torch
import torchvision
from torchvision import transforms as T

_model = None

def load_model():
    global _model
    if _model is None:
        # load torchvision pretrained object detector
        _model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        _model.eval()
    return _model


def detect_defects(image_path: str):
    """Run object detection on the provided image and return results.

    Args:
        image_path: path to the image file.

    Returns:
        dict with keys: 'image', 'boxes', 'scores', 'labels'
    """
    model = load_model()
    # load image via PIL
    from PIL import Image, ImageDraw
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    tensor = transform(img)
    with torch.no_grad():
        output = model([tensor])[0]
    boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    labels = output['labels'].cpu().numpy()

    results = []
    draw = ImageDraw.Draw(img)
    for box, score, label in zip(boxes, scores, labels):
        if score < 0.5:
            continue
        x1, y1, x2, y2 = box.tolist()
        results.append({
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2),
            'score': float(score),
            'label': int(label)
        })
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1-10), f"{int(label)}:{score:.2f}", fill="red")
    # convert to numpy array for display
    image_np = np.array(img)
    return {'image': image_np, 'boxes': results}
