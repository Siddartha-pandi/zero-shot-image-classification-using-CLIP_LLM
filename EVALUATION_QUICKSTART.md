 Evaluation Quick Reference

## Installation

```bash
cd backend
pip install matplotlib seaborn scikit-learn
```

## Common Commands

### Full Ablation Study
```bash
python evaluate.py \
  --dataset test_dataset.json \
  --mode full \
  --output results.json
```

### Individual Modes
```bash
# Baseline only
python evaluate.py --dataset test.json --mode baseline --output baseline.json

# Domain prompts only
python evaluate.py --dataset test.json --mode prompts --output prompts.json

# Adaptation tracking
python evaluate.py --dataset test.json --mode adaptation --output adaptation.json

# Online learning
python evaluate.py --dataset test.json --mode online --output online.json

# LLM reasoning (slow!)
python evaluate.py --dataset test.json --mode llm --output llm.json
```

### Generate Visualizations
```bash
python visualize_results.py \
  --results results.json \
  --output-dir evaluation_outputs
```

### Human Evaluation
```bash
# Generate template
python qualitative_eval.py \
  --action generate \
  --input results.json \
  --output human_eval.json \
  --samples 50

# Analyze completed evaluation
python qualitative_eval.py \
  --action analyze \
  --input human_eval_completed.json \
  --output human_analysis.json
```

## Dataset Format

```json
[
  {
    "path": "images/cat_001.jpg",
    "label": "cat",
    "domain": "natural"
  }
]
```

**Domains:** `natural`, `sketch`, `anime`, `medical`, `satellite`

## Typical Workflow

```bash
# 1. Prepare dataset
# Create test_dataset.json with your images

# 2. Run evaluation
python evaluate.py --dataset test_dataset.json --mode full --output results.json

# 3. Generate plots and reports
python visualize_results.py --results results.json --output-dir outputs

# 4. View results
cat outputs/EVALUATION_REPORT.md
```

## Output Files

### From evaluate.py
- `results.json` - All metrics in JSON format

### From visualize_results.py
- `ablation_comparison.png` - Bar chart
- `adaptation_curve.png` - Learning curve
- `domain_performance.png` - Per-domain accuracy
- `online_learning.png` - Before/after comparison
- `results_table.tex` - LaTeX table
- `EVALUATION_REPORT.md` - Full report

### From qualitative_eval.py
- `human_eval.json` - Template for ratings
- `human_analysis.json` - Statistics from ratings

## Key Metrics to Report

- **Top-1 Accuracy**: Main classification metric
- **Top-5 Accuracy**: Robustness indicator
- **Per-domain Accuracy**: Domain-specific performance
- **ECE**: Confidence calibration quality
- **Latency**: Speed (ms per image)
- **Online Learning Improvement**: Before/after accuracy gain
- **LLM Corrections**: Number of CLIP mistakes fixed

## Troubleshooting

**Import Error:** `pip install matplotlib seaborn scikit-learn`  
**Dataset Not Found:** Use absolute paths  
**Out of Memory:** Reduce dataset size or batch size  
**LLM Timeout:** Use smaller dataset for LLM mode  

## Performance Expectations

- Natural images: 70-85% top-1
- Sketch/anime: 60-75% top-1
- Medical/satellite: 50-70% top-1
- Online learning: +10-20% improvement
- LLM reasoning: +2-5% improvement

## For Your Report

Include:
1. Ablation study table (5 configs)
2. Plots (ablation, adaptation, domain)
3. Example predictions (3-5 with images)
4. Discussion (strengths, weaknesses, trade-offs)
