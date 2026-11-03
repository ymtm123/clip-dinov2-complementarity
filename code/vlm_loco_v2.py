"""
VLM-based LOCO AD logical anomaly detection v2.
Uses direct binary classification questions + reference image comparison.
"""
import os
import json
import re
import time
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

LOCO_ROOT = "data/MVTec_LOCO_AD"
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

# Normality specifications for each category (derived from training images)
NORMALITY_SPECS = {
    "breakfast_box": (
        "A normal breakfast box has two compartments:\n"
        "- LEFT compartment: 3 pieces of fruit (typically apples, peaches, mandarins, or tangerines)\n"
        "- RIGHT compartment: a mixture of granola/cereals, dried banana chips, and almonds\n"
        "All items should be present. No compartment should be empty or missing any type of item."
    ),
    "juice_bottle": (
        "A normal juice bottle has:\n"
        "- A TOP label on the upper part of the bottle\n"
        "- A BOTTOM label on the lower part of the bottle\n"
        "- Both labels should be present, undamaged, and in their correct positions (not swapped)"
    ),
    "pushpins": (
        "A normal pushpin arrangement has:\n"
        "- Groups of pushpins organized with separators between them\n"
        "- No extra pushpins beyond the expected arrangement\n"
        "- No missing pushpins from any group\n"
        "- All separators present between groups"
    ),
    "screw_bag": (
        "A normal screw bag contains:\n"
        "- Multiple screws in a transparent bag\n"
        "- All screws should be approximately the SAME LENGTH\n"
        "- No screws should be noticeably longer or shorter than the others"
    ),
    "splicing_connectors": (
        "Normal splicing connectors:\n"
        "- Each connector type matches the number of cables it connects\n"
        "- Cables are the appropriate length\n"
        "- Connector types are consistent and correctly matched to cable configurations"
    ),
}


def load_model():
    """Load Qwen2-VL with 4-bit quantization."""
    print(f"Loading {MODEL_NAME}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print("Model loaded successfully!")
    return model, processor


def score_image_binary(model, processor, image_path, category, ref_image_path=None):
    """Score an image by asking a direct normality question.
    Returns a score between 0 (normal) and 1 (anomalous).
    """
    spec = NORMALITY_SPECS[category]

    if ref_image_path:
        # Few-shot: provide a reference normal image
        prompt = (
            f"I will show you two images. The FIRST image is a reference showing a NORMAL {category.replace('_', ' ')}. "
            f"The SECOND image is a test image that may or may not be normal.\n\n"
            f"Normal specification:\n{spec}\n\n"
            f"Compare the test image to the reference. "
            f"Rate the test image's normality on a scale from 1 to 10, where:\n"
            f"- 1 means DEFINITELY ANOMALOUS (something is clearly wrong or missing)\n"
            f"- 10 means DEFINITELY NORMAL (everything matches the specification)\n\n"
            f"Reply ONLY with a single number (1-10) and a one-sentence explanation."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{ref_image_path}"},
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        # Zero-shot: only text specification
        prompt = (
            f"Look at this image of a {category.replace('_', ' ')}.\n\n"
            f"Normal specification:\n{spec}\n\n"
            f"Does this image match the normal specification? "
            f"Rate it on a scale from 1 to 10, where:\n"
            f"- 1 means DEFINITELY ANOMALOUS (something is clearly wrong, missing, or out of place)\n"
            f"- 10 means DEFINITELY NORMAL (everything matches the specification perfectly)\n\n"
            f"Reply ONLY with a single number (1-10) and a one-sentence explanation."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Extract numeric score from response
    numbers = re.findall(r'\b(\d+)\b', response)
    if numbers:
        raw_score = int(numbers[0])
        raw_score = max(1, min(10, raw_score))
        # Convert: 10=normal(0.0) to 1=anomalous(1.0)
        anomaly_score = (10 - raw_score) / 9.0
    else:
        anomaly_score = 0.5  # fallback

    return anomaly_score, response


def run_experiment(category, mode='zero_shot', max_images=None):
    """Run full experiment on one category."""
    print(f"\n{'='*60}")
    print(f"Category: {category} | Mode: {mode}")
    print(f"{'='*60}")

    model, processor = load_model()

    cat_dir = os.path.join(LOCO_ROOT, category)

    # Get reference image if few-shot
    ref_image = None
    if mode == 'one_shot':
        train_good_dir = os.path.join(cat_dir, "train", "good")
        train_images = sorted([f for f in os.listdir(train_good_dir) if f.endswith('.png')])
        ref_image = os.path.join(train_good_dir, train_images[0])
        print(f"Reference image: {ref_image}")

    # Collect test images
    test_images = []

    # Good images
    good_dir = os.path.join(cat_dir, "test", "good")
    for f in sorted(os.listdir(good_dir)):
        if f.endswith('.png'):
            test_images.append({
                'path': os.path.join(good_dir, f),
                'label': 0,
                'type': 'good'
            })

    # Logical anomalies
    logical_dir = os.path.join(cat_dir, "test", "logical_anomalies")
    for f in sorted(os.listdir(logical_dir)):
        if f.endswith('.png'):
            test_images.append({
                'path': os.path.join(logical_dir, f),
                'label': 1,
                'type': 'logical'
            })

    if max_images:
        good = [x for x in test_images if x['label'] == 0]
        anom = [x for x in test_images if x['label'] == 1]
        n = min(max_images // 2, len(good), len(anom))
        test_images = good[:n] + anom[:n]

    print(f"Testing {len(test_images)} images ({sum(1 for x in test_images if x['label']==0)} good, {sum(1 for x in test_images if x['label']==1)} anomalous)")

    gt_labels = []
    scores = []
    responses_log = []
    start_time = time.time()

    for i, img_info in enumerate(test_images):
        try:
            score, response = score_image_binary(
                model, processor, img_info['path'], category, ref_image
            )
        except Exception as e:
            print(f"  Error: {e}")
            score = 0.5
            response = f"ERROR: {str(e)}"

        gt_labels.append(img_info['label'])
        scores.append(score)
        responses_log.append({
            'path': os.path.basename(img_info['path']),
            'type': img_info['type'],
            'label': img_info['label'],
            'score': round(score, 4),
            'response': response[:300],
        })

        if i < 5 or (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(test_images)}] {img_info['type']:10s} | score={score:.3f} | {response[:60]}... | {rate:.1f} img/s")

    elapsed = time.time() - start_time

    # Compute AUROC
    gt_array = np.array(gt_labels)
    score_array = np.array(scores)
    if len(np.unique(gt_array)) == 2:
        auroc = roc_auc_score(gt_array, score_array)
    else:
        auroc = float('nan')

    print(f"\n{'='*40}")
    print(f"Results for {category} ({mode}):")
    print(f"  Image-level AUROC (logical only): {auroc*100:.1f}%")
    print(f"  N good: {sum(1 for l in gt_labels if l == 0)}, N anomalous: {sum(1 for l in gt_labels if l == 1)}")
    print(f"  Time: {elapsed:.1f}s ({len(test_images)/elapsed:.2f} img/s)")
    print(f"  Avg score (good): {np.mean([s for s,l in zip(scores, gt_labels) if l==0]):.3f}")
    print(f"  Avg score (anomalous): {np.mean([s for s,l in zip(scores, gt_labels) if l==1]):.3f}")
    print(f"{'='*40}")

    # Save results
    results_dir = "../results/vlm_loco_v2"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{category}_{mode}_results.json"), 'w') as f:
        json.dump({
            'category': category,
            'mode': mode,
            'auroc': float(auroc),
            'n_good': sum(1 for l in gt_labels if l == 0),
            'n_anomalous': sum(1 for l in gt_labels if l == 1),
            'time_seconds': elapsed,
            'avg_score_good': float(np.mean([s for s,l in zip(scores, gt_labels) if l==0])),
            'avg_score_anom': float(np.mean([s for s,l in zip(scores, gt_labels) if l==1])),
            'responses': responses_log
        }, f, indent=2)

    del model, processor
    torch.cuda.empty_cache()

    return auroc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="breakfast_box",
                       choices=list(NORMALITY_SPECS.keys()) + ['all'])
    parser.add_argument("--mode", type=str, default="zero_shot",
                       choices=["zero_shot", "one_shot"])
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    if args.category == 'all':
        categories = list(NORMALITY_SPECS.keys())
    else:
        categories = [args.category]

    results = {}
    for cat in categories:
        auroc = run_experiment(cat, args.mode, args.max_images)
        results[cat] = auroc

    # Print summary
    print(f"\n{'='*60}")
    print("Summary - VLM LOCO AD (logical anomalies)")
    print(f"{'='*60}")
    for cat, auroc in results.items():
        print(f"  {cat:25s}: {auroc*100:.1f}%")
    mean_auroc = np.mean(list(results.values()))
    print(f"  {'mean':25s}: {mean_auroc*100:.1f}%")
