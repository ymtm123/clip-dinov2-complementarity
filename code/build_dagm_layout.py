"""
Build a DAGM -> MVTec-style layout so the existing AnomalyCLIP / AnomalyDINO pipelines run on it.

DAGM_KaggleUpload/<ClassN>/{Train,Test}/*.PNG with defect masks in {Train,Test}/Label/<stem>_label.PNG.
A test image is ANOMALOUS iff it has a corresponding <stem>_label.PNG; otherwise NORMAL.

Produces (hardlinks where possible -> no data duplication; same volume; falls back to copy):
  <OUT>/<ClassN>/
     train/good/<stem>.PNG          (Train normal images = the 1-shot reference pool)
     test/good/<stem>.PNG           (Test normal images)
     test/defect/<stem>.PNG         (Test anomalous images)
     ground_truth/defect/<stem>.png (Test masks, renamed to match; for pixel-level later)

Image-level scoring uses only train/good + test/{good,defect}; ground_truth enables
pixel-level if pursued. Deterministic, resumable (skips files already linked).

Run with the default python:
  python src/build_dagm_layout.py
"""
import os
import shutil

SRC = "data/DAGM_2007/DAGM_KaggleUpload"
OUT = "data/DAGM_2007/dagm_mvtec"
CLASSES = [f"Class{i}" for i in range(1, 11)]


def _link(src, dst):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)          # hardlink (no copy, same volume)
    except OSError:
        shutil.copy2(src, dst)     # fallback


def anomalous_stems(phase_dir):
    label_dir = os.path.join(phase_dir, "Label")
    if not os.path.isdir(label_dir):
        return set()
    return {f.split("_")[0] for f in os.listdir(label_dir) if f.lower().endswith(".png")}


def build_class(cls):
    cdir = os.path.join(SRC, cls)
    out = os.path.join(OUT, cls)
    counts = dict(train_good=0, test_good=0, test_defect=0, masks=0)
    # train/good = Train normal images
    tr = os.path.join(cdir, "Train")
    anom_tr = anomalous_stems(tr)
    for f in sorted(os.listdir(tr)):
        if not f.lower().endswith(".png"):
            continue
        if os.path.splitext(f)[0] in anom_tr:
            continue
        _link(os.path.join(tr, f), os.path.join(out, "train", "good", f)); counts["train_good"] += 1
    # test/{good,defect} + ground_truth/defect
    te = os.path.join(cdir, "Test")
    anom_te = anomalous_stems(te)
    label_dir = os.path.join(te, "Label")
    for f in sorted(os.listdir(te)):
        if not f.lower().endswith(".png"):
            continue
        stem = os.path.splitext(f)[0]
        if stem in anom_te:
            _link(os.path.join(te, f), os.path.join(out, "test", "defect", f)); counts["test_defect"] += 1
            # mask: <stem>_label.PNG -> ground_truth/defect/<stem>.png
            msrc = os.path.join(label_dir, f"{stem}_label.PNG")
            if os.path.exists(msrc):
                _link(msrc, os.path.join(out, "ground_truth", "defect", f"{stem}.png")); counts["masks"] += 1
        else:
            _link(os.path.join(te, f), os.path.join(out, "test", "good", f)); counts["test_good"] += 1
    return counts


def main():
    print(f"Building DAGM -> MVTec layout under {OUT}")
    total = dict(train_good=0, test_good=0, test_defect=0, masks=0)
    for cls in CLASSES:
        c = build_class(cls)
        for k in total:
            total[k] += c[k]
        print(f"  {cls}: train/good={c['train_good']} test/good={c['test_good']} "
              f"test/defect={c['test_defect']} masks={c['masks']}")
    print(f"\n  TOTAL: train/good={total['train_good']} test/good={total['test_good']} "
          f"test/defect={total['test_defect']} masks={total['masks']}  (10 classes)")
    print(f"  Done: {OUT}")


if __name__ == "__main__":
    main()
