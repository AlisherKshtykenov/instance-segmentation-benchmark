"""Данные: аннотации COCO val2017, подвыборка из 500 фото и её скачивание."""
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{}"
N_IMAGES = 500
SEED = 42


def ann_path(root: Path) -> Path:
    return root / "data" / "raw" / "annotations" / "instances_val2017.json"


def ensure_annotations(root: Path) -> Path:
    """Скачивает архив аннотаций, если json ещё нет, и достаёт из него instances_val2017.json."""
    p = ann_path(root)
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    zip_path = p.parent.parent / "annotations_trainval2017.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(ANN_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extract("annotations/instances_val2017.json", p.parent.parent)
    zip_path.unlink()
    return p


def make_subset(root: Path, n: int = N_IMAGES, seed: int = SEED) -> Path:
    """Фиксирует подвыборку: n случайных фото из 5000 (random_state=seed) и пишет мини-аннотацию.

    Мини-аннотация содержит только выбранные фото и их объекты, поэтому COCOeval
    и вся статистика работают ровно на подвыборке.
    """
    out = root / "data" / "processed" / f"instances_val2017_subset{n}.json"
    if out.exists():
        return out
    full = json.loads(ensure_annotations(root).read_text())
    ids = pd.Series(sorted(im["id"] for im in full["images"]))
    chosen = set(ids.sample(n, random_state=seed).tolist())
    sub = {
        "info": full.get("info", {}),
        "licenses": full.get("licenses", []),
        "categories": full["categories"],
        "images": [im for im in full["images"] if im["id"] in chosen],
        "annotations": [a for a in full["annotations"] if a["image_id"] in chosen],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sub))
    return out


def image_dir(root: Path) -> Path:
    return root / "data" / "raw" / "val2017_subset"


def download_images(root: Path, subset_json: Path, workers: int = 16) -> Path:
    """Скачивает только фото подвыборки (около 80 МБ вместо 0.8 ГБ всего val2017)."""
    d = image_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    names = [im["file_name"] for im in json.loads(subset_json.read_text())["images"]]
    todo = [n for n in names if not (d / n).exists()]

    def fetch(name):
        tmp = d / (name + ".part")
        urllib.request.urlretrieve(IMG_URL.format(name), tmp)
        tmp.rename(d / name)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch, todo))
    return d


def subset_tables(subset_json: Path):
    """Таблицы фото и объектов подвыборки для EDA."""
    js = json.loads(subset_json.read_text())
    cats = {c["id"]: c["name"] for c in js["categories"]}
    imgs = pd.DataFrame(js["images"])[["id", "file_name", "width", "height"]]
    anns = pd.DataFrame(js["annotations"])
    anns["category"] = anns["category_id"].map(cats)
    # Деление на S/M/L в COCO идёт по площади маски: < 32², 32²–96², > 96²
    anns["size"] = pd.cut(anns["area"], [0, 32**2, 96**2, float("inf")],
                          labels=["small", "medium", "large"], right=False)
    return imgs, anns, cats
