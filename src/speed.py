"""Скорость: задержка на кадр и FPS при batch = 1.

Два режима:
- 'model': только сеть на уже подготовленном тензоре (без чтения фото, resize и сборки масок);
- 'full':  полный цикл, чтение фото с диска → предобработка → сеть → постобработка →
           бинарные маски в размере фото на CPU (как их получит приложение).

Перед каждой засечкой и после неё ждём GPU (synchronize), первые WARMUP кадров не считаем.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .models import load_mrcnn, load_yolo
from .predict import mrcnn_predict_one, read_rgb, sync, yolo_predict_one

WARMUP = 10
SPEED_CONF = 0.25  # рабочий порог для замеров скорости, одинаковый у всех моделей


def _summary(times_ms: list[float], warmup: int = WARMUP) -> dict:
    t = np.array(times_ms[warmup:])
    return {"ms_mean": t.mean(), "ms_std": t.std(), "ms_median": float(np.median(t)),
            "fps": 1000 / t.mean(), "n_timed": len(t)}


def _timed(fn, dev) -> float:
    sync(dev)
    t0 = time.perf_counter()
    fn()
    sync(dev)
    return (time.perf_counter() - t0) * 1000


@torch.no_grad()
def bench_mrcnn(root: Path, paths: list[Path], dev: str, half: bool, mode: str,
                min_size: int = 800) -> dict:
    model = load_mrcnn(root, min_size=min_size, score_thresh=SPEED_CONF).to(dev)
    if half:
        model = model.half()
    dtype = torch.float16 if half else torch.float32
    times = []
    for p in paths:
        if mode == "full":
            times.append(_timed(lambda: mrcnn_predict_one(model, read_rgb(p), dev, half), dev))
        else:
            # Предобработка (нормализация и resize) вне засечки
            x = torch.from_numpy(read_rgb(p)).to(dev).permute(2, 0, 1).to(dtype).div_(255)
            batch, _ = model.transform([x])

            def net():
                feats = model.backbone(batch.tensors)
                props, _ = model.rpn(batch, feats)
                model.roi_heads(feats, props, batch.image_sizes)  # маски 28×28 на объект
            times.append(_timed(net, dev))
    return _summary(times, WARMUP if dev != "cpu" else 5)


@torch.no_grad()
def bench_yolo(root: Path, name: str, paths: list[Path], dev: str, half: bool, mode: str,
               imgsz: int = 640) -> dict:
    yolo = load_yolo(root, name)  # свежий объект: у predictor Ultralytics своё состояние устройства
    # Первый вызов создаёт predictor с нужными device/half/imgsz
    yolo_predict_one(yolo, paths[0], dev, imgsz=imgsz, conf=SPEED_CONF, half=half)
    times = []
    if mode == "full":
        for p in paths:
            times.append(_timed(lambda: yolo_predict_one(
                yolo, p, dev, imgsz=imgsz, conf=SPEED_CONF, half=half), dev))
    else:
        pred = yolo.predictor
        net = pred.model  # AutoBackend: уже на устройстве и в нужной точности
        for p in paths:
            x = pred.preprocess([cv2.imread(str(p))])  # letterbox + нормализация вне засечки
            times.append(_timed(lambda: net(x), dev))
    return _summary(times, WARMUP if dev != "cpu" else 5)


def bench(root: Path, model_name: str, paths: list[Path], dev: str, half: bool, mode: str,
          size: int) -> dict:
    if model_name == "mrcnn":
        return bench_mrcnn(root, paths, dev, half, mode, min_size=size)
    return bench_yolo(root, model_name, paths, dev, half, mode, imgsz=size)
