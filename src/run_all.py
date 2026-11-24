"""Все тяжёлые прогоны одним запуском (кэш в results/): python -m src.run_all"""
import json
from pathlib import Path

from .data import download_images, image_dir, make_subset
from .experiments import accuracy_table, speed_table
from .models import passport

N_AP = 150  # AP на первых 150 фото подвыборки из 500 (ограничение по времени)

if __name__ == "__main__":
    import pandas as pd
    root = Path(__file__).resolve().parent.parent
    sj = make_subset(root)
    download_images(root, sj)
    images = json.loads(sj.read_text())["images"]
    print(accuracy_table(root, sj, images[:N_AP], image_dir(root)).round(1).to_string(), flush=True)
    print(speed_table(root, image_dir(root), images).to_string(), flush=True)
    pd.DataFrame(passport(root)).to_csv(root / "results" / "tables" / "passport.csv", index=False)
