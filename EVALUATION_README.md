# Comprehensive Evaluation Framework

A complete evaluation system for testing all three components of the zero-shot image classification project:
1. **Zero-shot classifier** (CLIP + domain-aware prompts)
2. **Adaptive system** (domain detection + auto-tuning + online learning)
3. **LLM layer** (reasoning + narrative generation)

## Overview

The evaluation framework provides:
- **Quantitative metrics**: Top-k accuracy, per-domain/class metrics, calibration, latency
- **Ablation study**: Tests 5 configurations to measure each component's contribution
- **Adaptation tracking**: Monitors accuracy improvements during online learning
- **Qualitative evaluation**: Human rating templates for LLM reasoning quality
- **Visualization**: Plots, confusion matrices, LaTeX tables, markdown reports

## File Structure

```
backend/
├── evaluate.py              # Main evaluation script (quantitative)
├── qualitative_eval.py      # Human evaluation tools (qualitative)
├── visualize_results.py     # Plotting and report generation
├── example_dataset.json     # Example dataset format
└── requirements.txt         # Add: matplotlib, seaborn, scikit-learn
```

---

## 1. Dataset Preparation

### Format

Create a JSON file with your test images:

```json
[
  {
    "path": "path/to/image.jpg",
    "label": "cat",
    "domain": "natural"
  },
  ...
]
```

**Fields:**
- `path`: Absolute or relative path to image file
- `label`: Ground truth class label (lowercase recommended)
- `domain`: One of: `natural`, `sketch`, `anime`, `medical`, `satellite`

### Recommendations

- **Size**: 20-50 images per class minimum
- **Diversity**: Include multiple domains
- **Balance**: Similar number of samples per class
- **Quality**: Representative of real-world use cases

### Example Dataset Structure

```
test_dataset/
├── natural/
│   ├── cat_001.jpg
│   ├── cat_002.jpg
│   ├── dog_001.jpg
│   └── car_001.jpg
├── sketch/
│   ├── cat_sketch_001.jpg
│   └── dog_sketch_001.jpg
├── anime/
│   └── character_001.jpg
├── medical/
│   └── xray_001.jpg
└── satellite/
    └── building_001.jpg
```

---

## 2. Running Evaluations

### Install Dependencies

```bash
cd backend
pip install matplotlib seaborn scikit-learn
```

### Full Ablation Study (Recommended)

Tests all 5 configurations:

```bash
python evaluate.py --dataset my_test_dataset.json --mode full --output results.json
```

This will run:
1. **CLIP Baseline** - Simple prompts, no adaptation
2. **+ Domain Prompts** - Domain-specific prompt templates
3. **+ Auto-Tuning** - Prototype adaptation during inference
4. **+ Online Learning** - Dynamic class addition
5. **+ LLM Reasoning** - Final system with reasoning layer

**Output:** JSON file with all metrics + console summary table

### Individual Modes

Run specific evaluations:

```bash
# Just baseline
python evaluate.py --dataset test.json --mode baseline

# Just domain prompts
python evaluate.py --dataset test.json --mode prompts

# Just adaptation tracking
python evaluate.py --dataset test.json --mode adaptation

# Just online learning
python evaluate.py --dataset test.json --mode online

# Just LLM reasoning
python evaluate.py --dataset test.json --mode llm
```

---

## 3. Understanding Results

### Sample Output

```
================================================================================
ABLATION STUDY SUMMARY
================================================================================
Configuration                            Top-1      Top-5      Latency (ms)
--------------------------------------------------------------------------------
1. CLIP only (baseline)                  68.0%      87.0%      120.5
2. + Domain-aware prompts                73.0%      90.0%      125.3
3. + Auto-tuning                         78.0%      92.0%      130.2
4. + Online learning                     84.0%      95.0%      135.1
5. + LLM reasoning                       86.0%      96.0%      850.4
================================================================================
```

### Key Metrics

**Accuracy Metrics:**
- `top1_accuracy`: Percentage of correct top predictions
- `top5_accuracy`: True label in top-5 candidates
- `domain_accuracy`: Per-domain breakdown
- `class_accuracy`: Per-class breakdown

**Advanced Metrics:**
- `expected_calibration_error`: Confidence calibration quality
- `avg_confidence`: Mean prediction confidence
- `avg_latency_ms`: Average processing time

**Online Learning:**
- `before_addition`: Accuracy on new classes before adding them
- `after_addition`: Accuracy after adding with 1-2 examples
- `improvement`: Absolute accuracy gain

**LLM Analysis:**
- `llm_improvements`: Number of times LLM corrected CLIP mistakes
- `improvement_examples`: Specific cases with reasoning

---

## 4. Visualization & Reports

### Generate All Visualizations

```bash
python visualize_results.py --results results.json --output-dir outputs
```

**Creates:**
- `ablation_comparison.png` - Bar chart comparing configurations
- `adaptation_curve.png` - Accuracy over time during auto-tuning
- `domain_performance.png` - Per-domain accuracy breakdown
- `online_learning.png` - Before/after class addition
- `confusion_matrix.png` - Classification confusion matrix (if enabled)
- `results_table.tex` - LaTeX table for papers/reports
- `EVALUATION_REPORT.md` - Comprehensive markdown report

### Example Plots

**Ablation Comparison:**
Shows progressive improvement from baseline to full system.

**Adaptation Curve:**
Demonstrates learning over time as prototypes adapt to new data.

**Domain Performance:**
Identifies which domains work well vs need improvement.

---

## 5. Qualitative Evaluation (LLM Reasoning)

### Generate Human Evaluation Template

```bash
python qualitative_eval.py --action generate --input results.json --output human_eval.json --samples 50
```

Creates a template with:
- 50 randomly selected samples
- Image paths and predictions
- Rating scales (1-5) for:
  - Reasoning quality
  - Narrative fluency
  - Narrative detail
  - Faithfulness (no hallucinations)

### Complete Human Evaluation

1. Open `human_eval.json`
2. View each image
3. Rate the LLM's reasoning and narrative
4. Fill in ratings and comments
5. Save as `human_eval_completed.json`

### Analyze Results

```bash
python qualitative_eval.py --action analyze --input human_eval_completed.json --output human_analysis.json
```

**Output:**
- Mean scores for each rating dimension
- Distribution of ratings
- Percentage of correct reasoning
- Summary statistics

---

## 6. For Your Project Report/Viva

### What to Include

**1. Quantitative Results Table**

Use the ablation study summary showing progressive improvements.

**2. Plots**

- Ablation bar chart (shows each component's contribution)
- Adaptation curve (demonstrates online learning)
- Domain performance (shows robustness across domains)

**3. Qualitative Examples**

Include 3-5 examples where:
- LLM corrected a CLIP mistake
- System worked well across domains
- Interesting reasoning/narrative generated

**4. Discussion Points**

- **Strengths**: Where system excels (e.g., natural images, common classes)
- **Weaknesses**: Failure cases (e.g., extreme domains, rare objects)
- **Trade-offs**: Accuracy vs latency (LLM adds 700ms but +2% accuracy)
- **Insights**: Which component contributed most (auto-tuning? prompts?)

### Sample Results Section

```markdown
## Results

### Ablation Study

We evaluated five progressive configurations:

| Configuration | Top-1 | Top-5 |
|--------------|-------|-------|
| CLIP Baseline | 68.0% | 87.0% |
| + Domain Prompts | 73.0% | 90.0% |
| + Auto-Tuning | 78.0% | 92.0% |
| + Online Learning | 84.0% | 95.0% |
| + LLM Reasoning | 86.0% | 96.0% |

Domain-aware prompts improved baseline by +5%, auto-tuning added +5%,
online learning contributed +6%, and LLM reasoning added final +2%.

### Online Learning

Adding 10 new classes with only 1-2 examples per class improved
accuracy on those classes from 4% to 78% without retraining.

### LLM Reasoning

The LLM corrected 45 CLIP mistakes by re-ranking candidates based on
visual reasoning. Human evaluation (n=50) rated reasoning quality at
4.2/5.0, with 88% mentioning correct visual cues.
```

---

## 7. Tips for Best Results

### Dataset Quality

✅ **Do:**
- Use diverse, representative images
- Include challenging cases (occlusions, unusual angles)
- Balance classes and domains
- Use clear, unambiguous labels

❌ **Don't:**
- Use low-resolution or corrupted images
- Mix very easy and very hard examples
- Have extreme class imbalance
- Use ambiguous or overlapping labels

### Interpretation

**High Top-1, High Top-5** → System works well  
**Low Top-1, High Top-5** → Correct answer in top-5, needs better ranking  
**Low Top-1, Low Top-5** → Fundamental classification failure  

**High ECE** → Overconfident or underconfident predictions  
**Low ECE** → Well-calibrated confidence scores  

**Large cross-domain drop** → Needs better domain adaptation  
**Small cross-domain drop** → Robust across domains  

### Performance Expectations

Realistic targets for zero-shot classification:
- **Natural images, common classes**: 70-85% top-1
- **Sketch/anime domains**: 60-75% top-1
- **Medical/satellite**: 50-70% top-1 (domain-specific)
- **With online learning**: +10-20% improvement
- **With LLM reasoning**: +2-5% improvement

---

## 8. Quick Start Example

```bash
# 1. Prepare your dataset
# Create test_dataset.json with your images

# 2. Run full evaluation
cd backend
python evaluate.py \
  --dataset test_dataset.json \
  --mode full \
  --output my_results.json

# 3. Generate visualizations
python visualize_results.py \
  --results my_results.json \
  --output-dir evaluation_outputs

# 4. Generate human eval template
python qualitative_eval.py \
  --action generate \
  --input my_results.json \
  --output human_eval.json \
  --samples 30

# 5. Complete human evaluation (manual)
# Fill in human_eval.json

# 6. Analyze human evaluation
python qualitative_eval.py \
  --action analyze \
  --input human_eval_completed.json \
  --output human_analysis.json
```

**Result:** Complete evaluation with quantitative metrics, plots, and qualitative analysis ready for your report!

---

## Troubleshooting

**Issue:** "FileNotFoundError: Dataset not found"  
**Fix:** Check dataset path is correct, use absolute paths if needed

**Issue:** "RuntimeError: No classes defined"  
**Fix:** Ensure class prototypes are initialized before evaluation

**Issue:** "LLM timeout/rate limit"  
**Fix:** Reduce dataset size or add delays between LLM calls

**Issue:** "Out of memory"  
**Fix:** Process images in batches, reduce batch size

**Issue:** "Import error: matplotlib"  
**Fix:** `pip install matplotlib seaborn scikit-learn`

---

## Advanced: Custom Metrics

You can extend `EvaluationMetrics` class to add:
- Per-class precision/recall/F1
- ROC curves and AUC
- Mean Average Precision (mAP)
- Custom domain-specific metrics

See `evaluate.py` for implementation details.

---

## Citation

If using this evaluation framework in your research, please cite your project:

```bibtex
@project{zero-shot-clip-llm,
  title={Zero-Shot Image Classification using CLIP and LLM},
  author={Your Name},
  year={2025},
  institution={Your Institution}
}
```
