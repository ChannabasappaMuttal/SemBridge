# SemBridge 🧥

**Semantic Bridge-Guided Training-Free Open-Vocabulary Segmentation for Fashion Visual Understanding**

***

## Overview

SemBridge is a **training-free**, zero-shot pipeline that segments fashion items from **free-form natural language prompts**. Unlike prior methods that require exact category labels, SemBridge accepts ambiguous, indirect, or descriptive inputs (e.g., *"Show me the arm covering"* → segments the sleeve).

### Architecture

![image_alt](https://github.com/ChannabasappaMuttal/SemBridge/blob/main/architecture_diagram.jpg?raw=true)

***

## Key Features

- **Zero-shot** — no fine-tuning required on any component
- **LLM-guided synonym expansion** — FLAN-T5-XL expands *"sleeve"* → *"sleeve. arm sleeve. long sleeve."*, dramatically improving Grounding DINO recall
- **Hybrid fallback parsing** — LLM → refinement → rule-based → TF-IDF → raw; never fails silently
- **Open-vocabulary attribute recognition** — CLIP predicts color, pattern, material, style, and detail from the segmented region
- **Dual dataset evaluation** — validated on both DeepFashion2 (13 coarse categories) and Fashionpedia (46 fine-grained categories + attributes)

***

## Datasets

### DeepFashion2
- **Script:** `SegBridge_deepfashion.py`
- 13 garment categories: `short_sleeved_shirt`, `long_sleeved_shirt`, `short_sleeved_outwear`, `long_sleeved_outwear`, `vest`, `sling`, `shorts`, `trousers`, `skirt`, `short_sleeved_dress`, `long_sleeved_dress`, `vest_dress`, `sling_dress`
- No attribute annotations; CLIP attribute prediction runs for qualitative richness only
- GT format: COCO-format JSON (`deepfashion2_val_coco.json`)

### Fashionpedia
- **Script:** `SegBridge_fashionpedia.py`
- 46 fine-grained categories including garments, garment parts, and accessories
- 294 attribute annotations (color, pattern, material, style, shape)
- GT format: native Fashionpedia JSON (compatible with COCO API)

***

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/ChannabasappaMuttal/SemBridge.git
cd SemBridge
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate
pip install opencv-python pillow matplotlib tqdm
pip install scikit-learn pycocotools
```

### 4. Install Segment Anything Model (SAM)
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 5. Download the SAM ViT-H checkpoint
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

***

## Pretrained Models (Auto-downloaded via HuggingFace Hub)

| Model | HuggingFace ID | Purpose |
|---|---|---|
| FLAN-T5-XL | `google/flan-t5-xl` | LLM parsing + synonym expansion |
| Grounding DINO | `IDEA-Research/grounding-dino-tiny` | Zero-shot detection |
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | Attribute recognition |
| SAM ViT-H | Manual download (see above) | Pixel-level segmentation |

All HuggingFace models download automatically on first run. Ensure ~20 GB free disk space and a CUDA-capable GPU (≥16 GB VRAM recommended).

***

## Usage

### DeepFashion2 Evaluation

```bash
python SegBridge_deepfashion.py \
  --json_path /path/to/deepfashion2_val_coco.json \
  --image_dir /path/to/deepfashion2/validation/image \
  --sam_checkpoint ./sam_vit_h_4b8939.pth \
  --num_images 200 \
  --run_ablation
```

### Fashionpedia Evaluation

```bash
python SegBridge_fashionpedia.py \
  --json_path /path/to/fashionpedia/instances_attributes_val2020.json \
  --image_dir /path/to/fashionpedia/val/ \
  --sam_checkpoint ./sam_vit_h_4b8939.pth \
  --num_images 200 \
  --run_ablation
```

### Single Image Inference (Quick Test)

```python
from SegBridge_deepfashion import FashionSegLLM

pipeline = FashionSegLLM(
    metadata_json="deepfashion2_val_coco.json",
    sam_checkpoint="./sam_vit_h_4b8939.pth",
)

masks = pipeline.run(
    image_path="your_image.jpg",
    user_prompt="Segment the long jacket",
    visualize=True,
    save_path="output.png"
)

for m in masks:
    print(f"  Label: {m['label']}")
    print(f"  CLIP Attributes: {m.get('visual_attributes', 'N/A')}")
```

***

## Output Files

After running evaluation, the following files are generated:

| File | Description |
|---|---|
| `eval_log.txt` | Full console log mirrored to disk |
| `df2_full_eval_results.json` | Per-image evaluation results (DeepFashion2) |
| `eval_summary.csv` | Overall metrics: mIoU, AP@0.5, Precision, Recall, F1 |
| `per_category.csv` | Per-category breakdown of mIoU and AP |
| `ablation_seed_42.json` | Ablation results — LLM vs Keyword-Only vs Raw-Text (seed 42) |
| `ablation_seed_123.json` | Ablation results (seed 123) |
| `ablation_seed_456.json` | Ablation results (seed 456) |
| `ablation_multiseed.json` | Aggregated multi-seed mean ± std |

***

## Ablation Study

The ablation compares three prompt parsing strategies across four prompt styles:

| Method | Strategy |
|---|---|
| **LLM-Parsed (Ours)** | FLAN-T5-XL with synonym expansion + hybrid fallback |
| **Keyword-Only** | Direct keyword matching, no synonym expansion |
| **Raw-Text** | Raw prompt passed directly to Grounding DINO |

Prompt styles evaluated: `simple`, `ambiguous`, `complex`, `indirect`

Run multi-seed ablation (seeds 42, 123, 456) for statistically robust results:

```python
from SegBridge_deepfashion import run_multi_seed_ablation, FashionSegLLM, FashionpediaGTLoader

pipeline = FashionSegLLM(metadata_json="...", sam_checkpoint="...")
gt_loader = FashionpediaGTLoader(annotation_json="...", image_dir="...")
run_multi_seed_ablation(pipeline, gt_loader, num_images=200, seeds=[42, 123, 456])
```

***

## Repository Structure

```
SemBridge/
├── SegBridge_deepfashion.py      # DeepFashion2 pipeline + evaluation
├── SegBridge_fashionpedia.py     # Fashionpedia pipeline + evaluation
├── README.md
├── sam_vit_h_4b8939.pth          # SAM checkpoint (download separately)
└── results/                      # Output JSON/CSV files (generated at runtime)
```

***

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 12 GB | 24 GB (A100/V100) |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB free | 50 GB (datasets + models) |
| CUDA | 11.7+ | 12.x |

CPU-only inference is supported but significantly slower (~10× per image).

***

## Citation

If you use SemBridge in your research, please cite:

```bibtex
@article{SemBridge2026,
  title   = {Semantic Bridge-Guided Training-Free Open-Vocabulary Fashion Segmentation},
  author  = {Muttal, Channabasappa and Giri, Chandadevi and Mulla, Md Naveed and Dhane, Ratan},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Manuscript submitted}
}
```

***

## Acknowledgements

This work builds on:
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) — Meta AI Research
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) — IDEA Research
- [CLIP](https://github.com/openai/CLIP) — OpenAI
- [FLAN-T5](https://huggingface.co/google/flan-t5-xl) — Google Research
- [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) dataset
- [Fashionpedia](https://fashionpedia.github.io/home/) dataset

***

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
