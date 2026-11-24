"""Все прогоны с кэшем: ноутбук вызывает эти функции, тяжёлое считается один раз.

Кэш: results/preds/*.json (предсказания) и results/tables/*_raw.csv (AP и замеры скорости).
Удалить файл = пересчитать эту часть.
"""
from pathlib import Path

import pandas as pd

from .evaluate import evaluate_config, load_gt
from .predict import device, run_config
from .speed import bench

# Главные конфигурации (как в документации моделей) и дополнительные для компромисса
MAIN = [("mrcnn", 800), ("YOLO26n-seg", 640), ("YOLO26s-seg", 640)]
TRADEOFF = [("mrcnn", 800),
            ("YOLO26n-seg", 320), ("YOLO26n-seg", 640), ("YOLO26n-seg", 960),
            ("YOLO26s-seg", 320), ("YOLO26s-seg", 640), ("YOLO26s-seg", 960)]
N_SPEED = 60  # одни и те же фото для всех моделей (60 ради времени прогона; полный замер: 200)
N_SPEED_CPU = 20  # на CPU Mask R-CNN идёт ~7 с на кадр


def label(model: str, size: int) -> str:
    if model == "mrcnn":
        return f"Mask R-CNN v2 (min_size {size})"
    return f"{model} (imgsz {size})"


def display_name(model: str) -> str:
    return "Mask R-CNN v2" if model == "mrcnn" else model


def accuracy_table(root: Path, subset_json: Path, images, img_dir, configs=TRADEOFF) -> pd.DataFrame:
    cache = root / "results" / "tables" / "ap_raw.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["model", "size"])
    gt = None
    rows = []
    for model, size in configs:
        hit = done[(done["model"] == model) & (done["size"] == size)]
        if len(hit):
            rows.append(hit.iloc[0].to_dict())
            continue
        preds = run_config(root, images, img_dir, model, size)
        gt = gt or load_gt(subset_json)
        rows.append({"model": model, "size": size,
                     **evaluate_config(gt, preds, [im["id"] for im in images])})
        pd.DataFrame(rows).to_csv(cache, index=False)  # сохраняем после каждой конфигурации
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def speed_runs(root: Path, img_dir: Path, images) -> list[dict]:
    """Список замеров: главные модели × (MPS FP32, MPS FP16, CPU) × (сеть, полный цикл),
    плюс полный цикл на MPS FP32 для всех конфигураций компромисса."""
    gpu = device()
    runs = []
    for model, size in MAIN:
        for dev, half in [(gpu, False), (gpu, True), ("cpu", False)]:
            for mode in ["model", "full"]:
                runs.append(dict(model=model, size=size, dev=dev, half=half, mode=mode))
    for model, size in TRADEOFF:
        if (model, size) not in MAIN:
            runs.append(dict(model=model, size=size, dev=gpu, half=False, mode="full"))
    return runs


def speed_table(root: Path, img_dir: Path, images) -> pd.DataFrame:
    cache = root / "results" / "tables" / "speed_raw.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame()
    key = ["model", "size", "dev", "half", "mode"]
    paths = [img_dir / im["file_name"] for im in images[:N_SPEED]]
    rows = []
    for r in speed_runs(root, img_dir, images):
        if len(done):
            hit = done[(done[key] == pd.Series(r)[key]).all(axis=1)]
            if len(hit):
                rows.append(hit.iloc[0].to_dict())
                continue
        n = N_SPEED_CPU if r["dev"] == "cpu" else N_SPEED
        try:
            res = bench(root, r["model"], paths[:n], r["dev"], r["half"], r["mode"], r["size"])
        except Exception as e:  # например, FP16 не поддержан на устройстве
            res = {"fps": float("nan"), "error": repr(e)[:200]}
        rows.append({**r, **res})
        print(r, f"{res['fps']:.1f} FPS")
        pd.DataFrame(rows).to_csv(cache, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df
