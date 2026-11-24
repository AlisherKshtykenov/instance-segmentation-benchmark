"""Загрузка моделей и их «паспорт»: число параметров и GFLOPs."""
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode
from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_V2_Weights,
                                          maskrcnn_resnet50_fpn_v2)

YOLO_NAMES = {"YOLO26n-seg": "yolo26n-seg.pt", "YOLO26s-seg": "yolo26s-seg.pt"}
MRCNN_NAME = "Mask R-CNN v2"


def load_mrcnn(root: Path, min_size: int = 800, score_thresh: float = 0.001):
    """Mask R-CNN ResNet-50-FPN v2 с весами COCO из torchvision.

    score_thresh=0.001 вместо 0.05 по умолчанию: для AP нужна вся кривая precision–recall.
    detections_per_img=100 совпадает с maxDets=100 в COCOeval.
    """
    torch.hub.set_dir(str(root / "weights" / "torch"))
    m = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
                                 min_size=min_size, box_score_thresh=score_thresh,
                                 detections_per_img=100)
    return m.eval()


def load_yolo(root: Path, name: str):
    """YOLO26-seg из Ultralytics; веса скачиваются в weights/ при первом запуске."""
    from ultralytics import YOLO
    w = root / "weights" / YOLO_NAMES[name]
    if not w.exists():
        YOLO(YOLO_NAMES[name])  # скачивает в текущую папку
        Path(YOLO_NAMES[name]).rename(w)
    return YOLO(str(w))


def n_params(module: torch.nn.Module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


def gflops(module: torch.nn.Module, x) -> float:
    """GFLOPs одного прохода на CPU, считает FlopCounterMode из PyTorch.

    FlopCounterMode считает FLOPs как 2 × MAC (умножение + сложение), так же
    считают Ultralytics и torchvision в своих таблицах.
    """
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        module(x)
    return fc.get_total_flops() / 1e9


def passport(root: Path) -> list[dict]:
    """Параметры (млн) и GFLOPs: у YOLO на входе 640×640, у Mask R-CNN на типичном фото COCO 800×1216 после resize."""
    rows = []
    m = load_mrcnn(root)
    # Эталон torchvision считает GFLOPs на 3×800×800 до resize; мы тоже даём 800×800,
    # внутренний transform оставит его 800×800.
    rows.append({"model": MRCNN_NAME, "params_m": n_params(m),
                 "gflops": gflops(m, [torch.rand(3, 800, 800)]), "input": "800×800"})
    for name in YOLO_NAMES:
        y = load_yolo(root, name).model.float().eval()
        rows.append({"model": name, "params_m": n_params(y),
                     "gflops": gflops(y, torch.rand(1, 3, 640, 640)), "input": "640×640"})
    return rows
