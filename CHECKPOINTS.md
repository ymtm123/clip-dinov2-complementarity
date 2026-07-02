# Model weights

All backbones are public; download from the official sources:

- CLIP ViT-L/14@336px — https://github.com/openai/CLIP
- DINOv2 ViT-L/14 (`dinov2_vitl14`) — https://github.com/facebookresearch/dinov2
- SigLIP ViT-L/16-384 (`vit_large_patch16_siglip_384`, timm) — https://huggingface.co/timm
- DINOv2 ViT-L/14 with registers (`dinov2_vitl14_reg`) — https://github.com/facebookresearch/dinov2
- MAE ViT-L/16 (`vit_large_patch16_224.mae`, timm) — https://huggingface.co/timm
- WideResNet-50-2 (torchvision ImageNet weights)
- Qwen2-VL-7B-Instruct (VLM logical-anomaly baseline) — https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

AnomalyCLIP object-agnostic prompts (learned on the auxiliary dataset, final epoch) are in `prompts/`:

- `9_12_4_multiscale_epoch_15.pth` — MVTec-trained prompts (used for VisA and BTAD)
- `9_12_4_multiscale_visa_epoch_15.pth` — VisA-trained prompts (used for MVTec AD)

AnomalyDINO uses frozen DINOv2 features with kNN scoring; it has no trained weights. EfficientAD student–teacher models are trained from scratch by `code/efficientad_benchmark_full.py` (anomalib).
