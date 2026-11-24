"""Предсказания обеих моделей в формате COCO results и их кэш в results/preds/."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from .models import load_mrcnn, load_yolo


def device() -> str:
    """CUDA нет (Mac M4), поэтому «GPU» здесь Apple MPS."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def sync(dev: str):
    """GPU асинхронный: перед засечкой времени ждём, пока он закончит работу."""
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def encode_mask(m: np.ndarray) -> dict:
    """Бинарная маска H×W → RLE. pycocotools ждёт uint8 в порядке Fortran, counts переводим в str для JSON."""
    rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def xyxy_to_xywh(b):
    """Рамки в COCO: [x, y, ширина, высота], а модели отдают углы [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = (float(v) for v in b)
    return [x1, y1, x2 - x1, y2 - y1]


def read_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


@torch.no_grad()
def mrcnn_predict_one(model, img_rgb: np.ndarray, dev: str, half: bool = False):
    """Один кадр через Mask R-CNN → (рамки xyxy, классы COCO id, score, маски H×W bool)."""
    x = torch.from_numpy(img_rgb).to(dev).permute(2, 0, 1).float().div_(255)
    if half:
        x = x.half()
    out = model([x])[0]
    # Маски приходят как вероятности 0…1 в размере исходного фото: порог 0.5
    masks = (out["masks"][:, 0] > 0.5).cpu().numpy()
    return (out["boxes"].float().cpu().numpy(), out["labels"].cpu().numpy(),
            out["scores"].float().cpu().numpy(), masks)


def yolo_predict_one(model, img_path, dev: str, imgsz: int = 640, conf: float = 0.001,
                     half: bool = False):
    """Один кадр через YOLO-seg. retina_masks=True даёт маски в размере исходного фото."""
    from ultralytics.data.converter import coco80_to_coco91_class
    r = model.predict(str(img_path), imgsz=imgsz, conf=conf, max_det=100, device=dev,
                      retina_masks=True, verbose=False, **({"half": True} if half else {}))[0]
    if r.boxes is None or len(r.boxes) == 0:
        h, w = r.orig_shape
        return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0), np.zeros((0, h, w), bool)
    # YOLO нумерует классы 0–79, а у COCO id 1–90 с пропусками
    to91 = np.array(coco80_to_coco91_class())
    labels = to91[r.boxes.cls.cpu().numpy().astype(int)]
    masks = (r.masks.data > 0.5).cpu().numpy()
    return r.boxes.xyxy.cpu().numpy(), labels, r.boxes.conf.cpu().numpy(), masks


def to_coco_results(image_id, boxes, labels, scores, masks) -> list[dict]:
    return [{"image_id": int(image_id), "category_id": int(c), "bbox": xyxy_to_xywh(b),
             "score": float(s), "segmentation": encode_mask(m)}
            for b, c, s, m in zip(boxes, labels, scores, masks)]


def run_config(root: Path, images: list[dict], img_dir: Path, model_name: str, size: int,
               dev: str | None = None, force: bool = False) -> Path:
    """Предсказания одной конфигурации на всех фото подвыборки, кэш в results/preds/<имя>.json.

    model_name: 'mrcnn' (size = min_size) или имя YOLO из models.YOLO_NAMES (size = imgsz).
    """
    dev = dev or device()
    tag = f"{model_name.replace(' ', '_')}_{size}"
    out = root / "results" / "preds" / f"{tag}.json"
    if out.exists() and not force:
        return out
    t0 = time.time()
    res = []
    if model_name == "mrcnn":
        model = load_mrcnn(root, min_size=size).to(dev)
        for im in images:
            res += to_coco_results(im["id"], *mrcnn_predict_one(
                model, read_rgb(img_dir / im["file_name"]), dev))
    else:
        model = load_yolo(root, model_name)
        for im in images:
            res += to_coco_results(im["id"], *yolo_predict_one(
                model, img_dir / im["file_name"], dev, imgsz=size))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print(f"{tag}: {len(res)} детекций, {time.time() - t0:.0f} с")
    return out
