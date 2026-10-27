"""
Build AnomalyCLIP meta.json for the DAGM->MVTec shim so test_save_scores.py can score it.
dataset.py reads {root}/meta.json -> meta[mode][cls_name] = [{img_path, mask_path, cls_name,
specie_name, anomaly}]. Paths are relative to root. Run with the default python.
"""
import os
import json

ROOT = "data/DAGM_2007/dagm_mvtec"
CLASSES = [f"Class{i}" for i in range(1, 11)]


def main():
    meta = {"train": {}, "test": {}}
    for cls in CLASSES:
        # test split
        rows = []
        good_dir = os.path.join(ROOT, cls, "test", "good")
        for f in sorted(os.listdir(good_dir)):
            if f.lower().endswith(".png"):
                rows.append(dict(img_path=f"{cls}/test/good/{f}", mask_path="", cls_name=cls,
                                 specie_name="good", anomaly=0))
        def_dir = os.path.join(ROOT, cls, "test", "defect")
        for f in sorted(os.listdir(def_dir)):
            if f.lower().endswith(".png"):
                stem = os.path.splitext(f)[0]
                rows.append(dict(img_path=f"{cls}/test/defect/{f}",
                                 mask_path=f"{cls}/ground_truth/defect/{stem}.png", cls_name=cls,
                                 specie_name="defect", anomaly=1))
        meta["test"][cls] = rows
        # train split (normal only; not used for zero-shot scoring but kept for completeness)
        tg = os.path.join(ROOT, cls, "train", "good")
        meta["train"][cls] = [dict(img_path=f"{cls}/train/good/{f}", mask_path="", cls_name=cls,
                                   specie_name="good", anomaly=0)
                              for f in sorted(os.listdir(tg)) if f.lower().endswith(".png")]
        print(f"  {cls}: test={len(meta['test'][cls])} (good+defect), train={len(meta['train'][cls])}")
    json.dump(meta, open(os.path.join(ROOT, "meta.json"), "w"), indent=2)
    print(f"  Saved {ROOT}/meta.json")


if __name__ == "__main__":
    main()
