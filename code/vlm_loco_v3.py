"""
VLM-based LOCO AD v3: Specific question-answer approach.
Ask targeted questions about countable/verifiable attributes,
then check if answers match expected values.
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

# Category-specific verification questions
# Each question has an expected answer pattern for NORMAL images
VERIFICATION_QUESTIONS = {
    "breakfast_box": [
        {
            "question": "Look at the LEFT compartment of this food tray. Count the number of individual fruits you see. Answer with ONLY a number.",
            "expected": "3",
            "check_fn": lambda ans: _extract_number(ans) == 3,
        },
        {
            "question": "Look at the RIGHT compartment of this food tray. Can you see ALL of these items: (1) granola or cereals, (2) banana chips, (3) almonds or nuts? Answer YES if all three types are present, NO if any are missing.",
            "expected": "YES",
            "check_fn": lambda ans: "yes" in ans.lower() and "no" not in ans.lower().split("yes")[0],
        },
        {
            "question": "Is the LEFT compartment of this food tray empty or does it contain fruit? Answer with FRUIT or EMPTY.",
            "expected": "FRUIT",
            "check_fn": lambda ans: "fruit" in ans.lower() or "peach" in ans.lower() or "apple" in ans.lower() or "mandarin" in ans.lower(),
        },
    ],
    "juice_bottle": [
        {
            "question": "Count the number of labels visible on this juice bottle. Answer with ONLY a number.",
            "expected": "2",
            "check_fn": lambda ans: _extract_number(ans) == 2,
        },
        {
            "question": "Does this juice bottle have a label on its UPPER/TOP half? Answer YES or NO only.",
            "expected": "YES",
            "check_fn": lambda ans: "yes" in ans.lower(),
        },
        {
            "question": "Does this juice bottle have a label on its LOWER/BOTTOM half? Answer YES or NO only.",
            "expected": "YES",
            "check_fn": lambda ans: "yes" in ans.lower(),
        },
        {
            "question": "Do any labels on this bottle appear damaged, torn, or partially missing? Answer YES or NO only.",
            "expected": "NO",
            "check_fn": lambda ans: "no" in ans.lower(),
        },
    ],
    # NOTE: pushpins / screw_bag / splicing_connectors questions redesigned A PRIORI from each category's
    # MVTec-LOCO logical-anomaly definition (count / composition), NOT tuned to the eval AUROC. The
    # original questions (saved baseline ~50%) were irrelevant to the count-based anomalies. Reported as-is.
    "pushpins": [
        {  # normal = 15 pushpins, exactly one in each of the 15 compartments
            "question": "This image shows a plastic box of compartments, each meant to hold exactly one yellow pushpin. Count the TOTAL number of pushpins visible. Answer with ONLY a number.",
            "expected": "15",
            "check_fn": lambda ans: _extract_number(ans) == 15,
        },
        {
            "question": "Does every compartment contain exactly ONE pushpin, with NO compartment empty and NONE holding two or more? Answer YES or NO.",
            "expected": "YES",
            "check_fn": lambda ans: "yes" in ans.lower() and "no" not in ans.lower().split("yes")[0],
        },
    ],
    "screw_bag": [
        {  # normal = 2 long screws + 2 short screws (4 screws), 2 hex nuts, 2 washers
            "question": "This bag contains screws, nuts, and washers. Count the total number of SCREWS (long threaded bolts). Answer with ONLY a number.",
            "expected": "4",
            "check_fn": lambda ans: _extract_number(ans) == 4,
        },
        {
            "question": "Count the number of HEX NUTS (six-sided metal nuts) in this bag. Answer with ONLY a number.",
            "expected": "2",
            "check_fn": lambda ans: _extract_number(ans) == 2,
        },
        {
            "question": "Count the number of WASHERS (flat metal rings) in this bag. Answer with ONLY a number.",
            "expected": "2",
            "check_fn": lambda ans: _extract_number(ans) == 2,
        },
    ],
    "splicing_connectors": [
        {  # normal = exactly two cables of the SAME color joined by one splice connector
            "question": "How many separate cables (wires) are joined by the splice connector in this image? Answer with ONLY a number.",
            "expected": "2",
            "check_fn": lambda ans: _extract_number(ans) == 2,
        },
        {
            "question": "Are the cables joined by the splice connector the SAME color as each other? Answer YES or NO.",
            "expected": "YES",
            "check_fn": lambda ans: "yes" in ans.lower() and "no" not in ans.lower().split("yes")[0],
        },
    ],
}


def _extract_number(text):
    """Extract first number from text."""
    nums = re.findall(r'\b(\d+)\b', text)
    return int(nums[0]) if nums else -1


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
    print("Model loaded!")
    return model, processor


def ask_vlm(model, processor, image_path, question):
    """Ask a single question about an image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": question},
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
        output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


def score_image(model, processor, image_path, category):
    """Score an image by checking multiple verification questions.
    Returns anomaly score [0, 1] and detailed results.
    """
    questions = VERIFICATION_QUESTIONS[category]
    n_failed = 0
    question_results = []

    for q_info in questions:
        response = ask_vlm(model, processor, image_path, q_info["question"])
        passed = q_info["check_fn"](response)
        if not passed:
            n_failed += 1
        question_results.append({
            "question": q_info["question"][:60],
            "expected": q_info["expected"],
            "response": response[:100],
            "passed": passed,
        })

    # Anomaly score = fraction of failed checks
    anomaly_score = n_failed / len(questions)
    return anomaly_score, question_results


def run_experiment(category, max_images=None, results_tag=""):
    """Run experiment on one category."""
    print(f"\n{'='*60}")
    print(f"Category: {category}")
    print(f"{'='*60}")

    model, processor = load_model()
    cat_dir = os.path.join(LOCO_ROOT, category)

    # Collect test images
    test_images = []
    good_dir = os.path.join(cat_dir, "test", "good")
    for f in sorted(os.listdir(good_dir)):
        if f.endswith('.png'):
            test_images.append({'path': os.path.join(good_dir, f), 'label': 0, 'type': 'good'})

    logical_dir = os.path.join(cat_dir, "test", "logical_anomalies")
    for f in sorted(os.listdir(logical_dir)):
        if f.endswith('.png'):
            test_images.append({'path': os.path.join(logical_dir, f), 'label': 1, 'type': 'logical'})

    if max_images:
        good = [x for x in test_images if x['label'] == 0]
        anom = [x for x in test_images if x['label'] == 1]
        n = min(max_images // 2, len(good), len(anom))
        test_images = good[:n] + anom[:n]

    n_questions = len(VERIFICATION_QUESTIONS[category])
    print(f"Testing {len(test_images)} images ({n_questions} questions each)")

    gt_labels = []
    scores = []
    all_results = []
    start_time = time.time()

    for i, img_info in enumerate(test_images):
        try:
            score, q_results = score_image(model, processor, img_info['path'], category)
        except Exception as e:
            print(f"  Error: {e}")
            score = 0.5
            q_results = [{"error": str(e)}]

        gt_labels.append(img_info['label'])
        scores.append(score)
        all_results.append({
            'path': os.path.basename(img_info['path']),
            'type': img_info['type'],
            'label': img_info['label'],
            'score': round(score, 4),
            'questions': q_results,
        })

        passed_str = "".join(["O" if q.get("passed", False) else "X" for q in q_results])
        if i < 6 or (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  [{i+1}/{len(test_images)}] {img_info['type']:10s} | score={score:.3f} | checks=[{passed_str}] | {(i+1)/elapsed:.2f} img/s")

    elapsed = time.time() - start_time
    gt_array = np.array(gt_labels)
    score_array = np.array(scores)
    auroc = roc_auc_score(gt_array, score_array) if len(np.unique(gt_array)) == 2 else float('nan')

    print(f"\n{'='*40}")
    print(f"Results for {category}:")
    print(f"  Image-level AUROC (logical): {auroc*100:.1f}%")
    print(f"  Avg score (good):  {np.mean([s for s,l in zip(scores, gt_labels) if l==0]):.3f}")
    print(f"  Avg score (anom):  {np.mean([s for s,l in zip(scores, gt_labels) if l==1]):.3f}")
    print(f"  Time: {elapsed:.1f}s ({len(test_images)/elapsed:.2f} img/s)")
    print(f"{'='*40}")

    # Save
    results_dir = f"../results/vlm_loco_v3{results_tag}"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{category}_results.json"), 'w') as f:
        json.dump({
            'category': category,
            'auroc': float(auroc),
            'time_seconds': elapsed,
            'n_images': len(test_images),
            'avg_score_good': float(np.mean([s for s,l in zip(scores, gt_labels) if l==0])),
            'avg_score_anom': float(np.mean([s for s,l in zip(scores, gt_labels) if l==1])),
            'details': all_results,
        }, f, indent=2, ensure_ascii=False)

    del model, processor
    torch.cuda.empty_cache()
    return auroc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="breakfast_box",
                       choices=list(VERIFICATION_QUESTIONS.keys()) + ['all'])
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--results_tag", type=str, default="",
                        help="suffix for the results dir (e.g. _apriori) to preserve the baseline run")
    args = parser.parse_args()

    if args.category == 'all':
        categories = list(VERIFICATION_QUESTIONS.keys())
    else:
        categories = [args.category]

    results = {}
    for cat in categories:
        auroc = run_experiment(cat, args.max_images, results_tag=args.results_tag)
        results[cat] = auroc

    print(f"\n{'='*60}")
    print("Summary - VLM LOCO AD v3 (verification questions)")
    print(f"{'='*60}")
    for cat, auroc in results.items():
        print(f"  {cat:25s}: {auroc*100:.1f}%")
    if len(results) > 1:
        print(f"  {'mean':25s}: {np.mean(list(results.values()))*100:.1f}%")
