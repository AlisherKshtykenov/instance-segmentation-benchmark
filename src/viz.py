"""Наложение масок на фото и выбор сложных примеров для сравнения глазами."""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from .predict import read_rgb

SHOW_SCORE = 0.5  # для картинок показываем только уверенные детекции


def _palette(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(40, 255, size=(max(n, 1), 3))


def overlay(img: np.ndarray, masks: list[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    """Каждая маска своим цветом плюс белый контур: так видно края и перекрытия."""
    out = img.astype(np.float32).copy()
    for m, c in zip(masks, _palette(len(masks))):
        out[m] = out[m] * (1 - alpha) + c * alpha
    out = out.astype(np.uint8)
    for m in masks:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, (255, 255, 255), 1)
    return out


def gt_masks(gt, image_id) -> list[np.ndarray]:
    anns = gt.loadAnns(gt.getAnnIds(imgIds=image_id, iscrowd=False))
    return [gt.annToMask(a).astype(bool) for a in anns]


def pred_masks(preds: list[dict], image_id, thr: float = SHOW_SCORE) -> list[np.ndarray]:
    return [mask_utils.decode(p["segmentation"]).astype(bool)
            for p in preds if p["image_id"] == image_id and p["score"] >= thr]


def pick_hard_images(anns: pd.DataFrame, k: int = 5) -> list[int]:
    """Фото с мелкими объектами и сложными границами.

    Сложность границы оцениваем числом вершин полигона на единицу периметра рамки:
    у тонких и изрезанных объектов (люди, велосипеды, ветки) вершин больше.
    Счёт = число мелких объектов + число объектов со сложной границей, без толп (iscrowd).
    """
    a = anns[anns["iscrowd"] == 0].copy()
    a["n_vertices"] = a["segmentation"].apply(lambda s: sum(len(p) // 2 for p in s))
    a["complex"] = a["n_vertices"] / (2 * (a["bbox"].str[2] + a["bbox"].str[3]) + 1e-6) > 0.25
    g = a.groupby("image_id").agg(n=("id", "size"), small=("size", lambda s: (s == "small").sum()),
                                  complex=("complex", "sum"))
    g = g[(g["n"] >= 6) & (g["n"] <= 25)]  # не пустые и не каша, где ничего не разглядеть
    g["score"] = g["small"] + g["complex"]
    return g.sort_values("score", ascending=False).head(k).index.tolist()


def compare_figure(gt, img_dir: Path, image_id, preds_by_model: dict, out_path: Path):
    """Ряд: эталон | модель 1 | модель 2 ... для одного фото."""
    info = gt.loadImgs(image_id)[0]
    img = read_rgb(img_dir / info["file_name"])
    panels = {"Эталон (COCO)": gt_masks(gt, image_id)}
    panels.update({name: pred_masks(p, image_id) for name, p in preds_by_model.items()})
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4))
    for ax, (title, ms) in zip(axes, panels.items()):
        ax.imshow(overlay(img, ms))
        ax.set_title(f"{title}: {len(ms)} масок")
        ax.axis("off")
    fig.suptitle(f"image_id {image_id} ({info['file_name']})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    return fig


def load_preds(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())
