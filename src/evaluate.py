"""Точность: COCOeval из pycocotools для рамок (bbox) и масок (segm)."""
import contextlib
import io
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Порядок чисел в COCOeval.stats (первые 6 из 12)
STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl"]


def load_gt(subset_json: Path) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(subset_json))


def coco_ap(gt: COCO, preds_json: Path, iou_type: str, img_ids=None) -> dict:
    """AP по COCO (среднее по IoU 0.50:0.05:0.95) и его разбивка; числа в процентах."""
    preds = json.loads(preds_json.read_text())
    if iou_type == "bbox":
        # Для рамок маски не нужны; без них loadRes работает быстрее
        preds = [{k: v for k, v in p.items() if k != "segmentation"} for p in preds]
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(preds)
        ev = COCOeval(gt, dt, iou_type)
        # Только фото, на которых запускали модели: иначе объекты остальных фото
        # засчитываются как пропущенные и AP занижается
        ev.params.imgIds = sorted(img_ids or gt.getImgIds())
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {n: 100 * v for n, v in zip(STAT_NAMES, ev.stats[:6])}


def evaluate_config(gt: COCO, preds_json: Path, img_ids=None) -> dict:
    """Строка таблицы: AP mask и AP box, плюс AP50/AP75/APs/APm/APl по маскам и по рамкам."""
    segm = coco_ap(gt, preds_json, "segm", img_ids)
    bbox = coco_ap(gt, preds_json, "bbox", img_ids)
    row = {"AP mask": segm["AP"], "AP box": bbox["AP"]}
    row.update({f"{k} mask": v for k, v in segm.items() if k != "AP"})
    row.update({f"{k} box": v for k, v in bbox.items() if k != "AP"})
    return row
