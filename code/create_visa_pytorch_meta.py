"""Create meta.json for VisA_pytorch/1cls format compatible with AnomalyCLIP."""
import json
import os
import glob

data_root = "data/VisA/VisA_pytorch/1cls"

categories = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
              'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

meta = {"train": {}, "test": {}}

for cat in categories:
    meta["train"][cat] = []
    meta["test"][cat] = []

    # Train: good images only
    train_good_dir = os.path.join(data_root, cat, "train", "good")
    if os.path.exists(train_good_dir):
        for img_name in sorted(os.listdir(train_good_dir)):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                meta["train"][cat].append({
                    "img_path": f"{cat}/train/good/{img_name}",
                    "mask_path": "",
                    "cls_name": cat,
                    "specie_name": "",
                    "anomaly": 0
                })

    # Test: good images
    test_good_dir = os.path.join(data_root, cat, "test", "good")
    if os.path.exists(test_good_dir):
        for img_name in sorted(os.listdir(test_good_dir)):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                meta["test"][cat].append({
                    "img_path": f"{cat}/test/good/{img_name}",
                    "mask_path": "",
                    "cls_name": cat,
                    "specie_name": "good",
                    "anomaly": 0
                })

    # Test: bad images
    test_bad_dir = os.path.join(data_root, cat, "test", "bad")
    if os.path.exists(test_bad_dir):
        for img_name in sorted(os.listdir(test_bad_dir)):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                meta["test"][cat].append({
                    "img_path": f"{cat}/test/bad/{img_name}",
                    "mask_path": f"{cat}/ground_truth/bad/{os.path.splitext(img_name)[0]}.png",
                    "cls_name": cat,
                    "specie_name": "bad",
                    "anomaly": 1
                })

    print(f"{cat}: train={len(meta['train'][cat])}, test={len(meta['test'][cat])} "
          f"(good={len([x for x in meta['test'][cat] if x['anomaly']==0])}, "
          f"bad={len([x for x in meta['test'][cat] if x['anomaly']==1])})")

output_path = os.path.join(data_root, "meta.json")
with open(output_path, 'w') as f:
    json.dump(meta, f, indent=2)
print(f"\nSaved to {output_path}")
print(f"Total test samples: {sum(len(meta['test'][c]) for c in categories)}")
