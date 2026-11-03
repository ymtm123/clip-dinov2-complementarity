"""
Pilot experiment: Local VLM (Qwen2-VL) for LOCO AD logical anomaly detection.
Tests whether a local VLM can detect logical anomalies in LOCO AD images
by asking category-specific questions about the image content.
"""
import os
import json
import time
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ===== Configuration =====
LOCO_ROOT = "data/MVTec_LOCO_AD"
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"  # 7B model for RTX 4090

# Category-specific questions for logical anomaly detection
CATEGORY_QUESTIONS = {
    "breakfast_box": {
        "description": "A food tray with two compartments. Left side has fruits (apple/peach/mandarin/tangerine). Right side has granola/almonds/banana chips.",
        "questions": [
            "Describe what you see in this breakfast box image. List all items visible in the left compartment and the right compartment separately.",
            "Are there any missing items or unusual arrangements in this breakfast box? Specifically, check: (1) fruits on the left side, (2) granola/cereals on the right, (3) almonds on the right, (4) banana chips on the right.",
        ]
    },
    "juice_bottle": {
        "description": "A juice bottle with two labels (top and bottom). Labels should be in correct position.",
        "questions": [
            "Describe this juice bottle. Focus on the labels - how many labels are there, and where are they positioned?",
            "Is there anything unusual about this juice bottle? Check: (1) Is the top label present? (2) Is the bottom label present? (3) Are labels in the correct position? (4) Are labels damaged?",
        ]
    },
    "pushpins": {
        "description": "An organized display of pushpins with separators between groups.",
        "questions": [
            "Describe what you see in this image of pushpins. How many pushpins are there, and how are they arranged?",
            "Is there anything unusual about the pushpin arrangement? Check: (1) Are there any extra pushpins? (2) Are any pushpins missing? (3) Are all separators present?",
        ]
    },
    "screw_bag": {
        "description": "A transparent bag containing screws of similar size.",
        "questions": [
            "Describe the screws in this bag. Are they all the same size? Count approximately how many screws you see.",
            "Is there anything unusual about the screws? Check: (1) Are any screws noticeably longer than others? (2) Are any screws noticeably shorter? (3) Are all screws approximately the same size?",
        ]
    },
    "splicing_connectors": {
        "description": "Electrical splicing connectors with cables. Different connector types for different numbers of cables.",
        "questions": [
            "Describe the splicing connectors and cables in this image. What types of connectors do you see?",
            "Is there anything unusual about the connectors? Check: (1) Are the correct connector types used for the number of cables? (2) Are the cables the correct length? (3) Are connectors properly matched?",
        ]
    },
}


def load_model():
    """Load Qwen2-VL with 4-bit quantization for RTX 4090."""
    print(f"Loading {MODEL_NAME}...")
    from transformers import BitsAndBytesConfig

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


def ask_vlm(model, processor, image_path, question):
    """Ask VLM a question about an image."""
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
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    # Decode only the generated tokens
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def detect_anomaly_from_response(response):
    """Simple anomaly scoring from VLM response."""
    response_lower = response.lower()

    # Keywords indicating anomaly
    anomaly_keywords = [
        "missing", "absent", "not present", "no ", "unusual", "different",
        "wrong", "incorrect", "damaged", "broken", "extra", "additional",
        "too long", "too short", "swapped", "out of place", "anomal",
        "irregular", "abnormal", "defect", "fewer", "more than",
    ]

    # Keywords indicating normal
    normal_keywords = [
        "normal", "everything looks", "no issues", "all present",
        "correctly", "proper", "as expected", "nothing unusual",
        "consistent", "uniform", "standard", "typical",
    ]

    anomaly_score = sum(1 for kw in anomaly_keywords if kw in response_lower)
    normal_score = sum(1 for kw in normal_keywords if kw in response_lower)

    # Score: higher = more likely anomalous
    score = anomaly_score - normal_score * 0.5
    return max(0, min(1, (score + 2) / 6))  # Normalize to [0, 1]


def run_pilot(category, max_images=None):
    """Run pilot experiment on one category."""
    print(f"\n{'='*60}")
    print(f"Category: {category}")
    print(f"{'='*60}")

    model, processor = load_model()

    cat_dir = os.path.join(LOCO_ROOT, category)
    questions_info = CATEGORY_QUESTIONS[category]
    question = questions_info["questions"][1]  # Use the anomaly detection question

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
        # Sample a balanced subset
        good = [x for x in test_images if x['label'] == 0]
        anom = [x for x in test_images if x['label'] == 1]
        n = min(max_images // 2, len(good), len(anom))
        test_images = good[:n] + anom[:n]

    print(f"Testing {len(test_images)} images ({sum(1 for x in test_images if x['label']==0)} good, {sum(1 for x in test_images if x['label']==1)} anomalous)")
    print(f"Question: {question[:80]}...")

    gt_labels = []
    scores = []
    responses_log = []

    for i, img_info in enumerate(test_images):
        try:
            response = ask_vlm(model, processor, img_info['path'], question)
            score = detect_anomaly_from_response(response)
        except Exception as e:
            print(f"Error on {img_info['path']}: {e}")
            response = "ERROR"
            score = 0.5

        gt_labels.append(img_info['label'])
        scores.append(score)
        responses_log.append({
            'path': os.path.basename(img_info['path']),
            'type': img_info['type'],
            'label': img_info['label'],
            'score': score,
            'response': response[:200],
        })

        if i < 5 or i % 10 == 0:
            print(f"  [{i+1}/{len(test_images)}] {img_info['type']:10s} | score={score:.3f} | {response[:80]}...")

    # Compute AUROC
    gt_array = np.array(gt_labels)
    score_array = np.array(scores)
    if len(np.unique(gt_array)) == 2:
        auroc = roc_auc_score(gt_array, score_array)
    else:
        auroc = float('nan')

    print(f"\n{'='*40}")
    print(f"Results for {category}:")
    print(f"  Image-level AUROC (logical only): {auroc*100:.1f}%")
    print(f"  N good: {sum(1 for l in gt_labels if l == 0)}")
    print(f"  N anomalous: {sum(1 for l in gt_labels if l == 1)}")
    print(f"{'='*40}")

    # Save detailed results
    results_dir = "../results/vlm_loco_pilot"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{category}_results.json"), 'w') as f:
        json.dump({
            'category': category,
            'auroc': float(auroc),
            'n_good': sum(1 for l in gt_labels if l == 0),
            'n_anomalous': sum(1 for l in gt_labels if l == 1),
            'responses': responses_log
        }, f, indent=2)

    # Clean up GPU memory
    del model, processor
    torch.cuda.empty_cache()

    return auroc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="breakfast_box",
                       choices=list(CATEGORY_QUESTIONS.keys()))
    parser.add_argument("--max_images", type=int, default=20,
                       help="Max images to test (None=all)")
    args = parser.parse_args()

    auroc = run_pilot(args.category, args.max_images)
    print(f"\nFinal AUROC: {auroc*100:.1f}%")
