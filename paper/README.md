# Research Paper: Domain-Adaptive Zero-Shot Image Classification

This directory contains the LaTeX source for the research paper on domain-adaptive zero-shot image classification using auto-tuned CLIP and large language models.

## Files

- `research_paper.tex` - Main paper LaTeX source
- `README.md` - This file
- `figures/` - Directory for figures and plots (to be created)

## Compilation Instructions

### Prerequisites

Install a LaTeX distribution:
- **Windows**: MiKTeX or TeX Live
- **macOS**: MacTeX
- **Linux**: TeX Live

### Compiling the Paper

```bash
# Navigate to the paper directory
cd "s:\Siddu\Final Year\zero-shot\paper"

# Compile (run twice for references)
pdflatex research_paper.tex
pdflatex research_paper.tex
```

Or use an online LaTeX editor like Overleaf.

## Customization Guide

### 1. Update Author Information

Replace the author block (lines 18-22) with your actual information:

```latex
\author{
    \IEEEauthorblockN{Your Full Name\IEEEauthorrefmark{1}}
    \IEEEauthorblockA{\IEEEauthorrefmark{1}Department of Computer Science\\
    Your University Name\\
    Email: your.email@university.edu}
}
```

### 2. Add Experimental Results

You need to run experiments and populate the tables with actual results:

#### Table 1 (Main Results) - Line 442
- Run evaluation on ImageNet, ChestX-ray, EuroSAT, Oxford Pets, Food-101
- Record top-1 accuracy for each method
- Update the table with your actual numbers

#### Table 2 (Ablation Components) - Line 473
- Run ablation study disabling each component
- Record accuracy changes
- Update the table

#### Table 3 (Domain Analysis) - Line 497
- Compare generic vs domain-specific prompts
- Run on each domain separately
- Update the table

#### Table 4 (Computational Efficiency) - Line 528
- Measure latency and memory usage
- Use GPU monitoring tools
- Update the table

### 3. Generate Figures

Create the following figures and save them in `figures/` directory:

#### Figure 1: System Architecture (manual diagram)
Create a conceptual diagram showing:
- Input image flow
- CLIP encoding
- Domain-aware prompts
- Adaptive prototype learning
- LLM reasoning
- Final prediction

Tools: Draw.io, PowerPoint, TikZ, or Inkscape

#### Figure 2: Adaptation Curve
```python
# Use backend/visualize_results.py or create custom plot
import matplotlib.pyplot as plt
import numpy as np

# Your data from experiments
samples = [0, 50, 100, 200, 300, 500]
accuracy_natural = [68.3, 70.1, 71.2, 72.0, 72.5, 73.2]
accuracy_medical = [42.1, 45.3, 47.8, 49.5, 50.8, 52.7]
# ... add other domains

plt.figure(figsize=(8, 5))
plt.plot(samples, accuracy_natural, marker='o', label='Natural')
plt.plot(samples, accuracy_medical, marker='s', label='Medical')
# ... plot other domains
plt.xlabel('Number of Adaptation Samples')
plt.ylabel('Top-1 Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('figures/adaptation_curve.pdf', bbox_inches='tight')
```

#### Figure 3: Hyperparameter Sensitivity
Create plots showing accuracy vs:
- Learning rate α (0.01 to 0.5)
- Confidence threshold τ (0.05 to 0.5)

#### Figure 4: Qualitative Examples
Create a multi-panel figure showing:
- Example images from each domain
- Top-5 predictions
- LLM reasoning explanation
- Ground truth labels

### 4. Run Experiments to Collect Data

Use the evaluation framework:

```bash
cd backend

# Run baseline CLIP evaluation
python evaluate.py --dataset imagenet --method baseline

# Run your method
python evaluate.py --dataset imagenet --method adaptive

# Run ablation studies
python evaluate.py --dataset imagenet --ablation domain_prompts
python evaluate.py --dataset imagenet --ablation adaptive_learning
python evaluate.py --dataset imagenet --ablation llm_reasoning

# Measure computational efficiency
python evaluate.py --dataset imagenet --measure-latency
```

### 5. Update Dataset Descriptions (Section 4.1)

Add specific details about your datasets:
- Number of samples
- Number of classes
- Train/test splits
- Data sources

### 6. Add Acknowledgments

Update the Acknowledgments section with:
- Funding sources
- Institutional support
- Dataset providers
- Collaborators

### 7. Customize for Target Journal

The current template uses IEEEtran style. Adjust for your target journal:

**For IEEE Transactions:**
- Keep current format
- Check specific journal guidelines

**For ACM journals:**
```latex
\documentclass[acmtog]{acmart}
```

**For CVPR/ICCV/ECCV:**
```latex
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage{cvpr}
```

**For NeurIPS/ICML:**
```latex
\documentclass{article}
\usepackage{neurips_2024}
```

## Paper Structure

1. **Abstract** (200-250 words) - ✓ Complete
2. **Introduction** (2-3 pages) - ✓ Complete
3. **Related Work** (2-3 pages) - ✓ Complete
4. **Methodology** (4-5 pages) - ✓ Complete
5. **Experiments** (1-2 pages) - ⚠️ Needs actual experimental data
6. **Results** (3-4 pages) - ⚠️ Needs actual results and figures
7. **Discussion** (1-2 pages) - ✓ Complete
8. **Conclusion** (1 page) - ✓ Complete

**Total Expected Length:** 12-16 pages (IEEE format)

## Checklist Before Submission

- [ ] Replace all placeholder author information
- [ ] Add all actual experimental results to tables
- [ ] Generate all figures (architecture, plots, examples)
- [ ] Run spell check and grammar check
- [ ] Verify all citations are complete
- [ ] Check equation formatting
- [ ] Ensure all figures have captions
- [ ] Verify table formatting
- [ ] Add acknowledgments
- [ ] Review journal-specific guidelines
- [ ] Get feedback from advisors/colleagues
- [ ] Check page limits
- [ ] Verify figure quality (min 300 DPI)
- [ ] Proofread multiple times

## Tips for Strong Paper

1. **Clear Contributions:** Highlight novelty in intro
2. **Strong Baselines:** Compare to recent SOTA methods
3. **Ablation Studies:** Show each component matters
4. **Error Analysis:** Discuss failure cases honestly
5. **Reproducibility:** Provide code, datasets, hyperparameters
6. **Visual Results:** Include qualitative examples
7. **Statistical Significance:** Report std dev, run multiple seeds
8. **Computational Cost:** Compare runtime and memory

## Additional Resources

- **LaTeX Help:** https://www.overleaf.com/learn
- **IEEE Author Center:** https://journals.ieeeauthorcenter.ieee.org/
- **Academic Writing Guide:** https://www.nature.com/nature/for-authors/formatting-guide
- **Figure Design:** "Ten Simple Rules for Better Figures" (PLOS Comp Bio)

## Contact

For questions about the implementation or paper, contact:
[Your contact information]

## Citation

If you use this work, please cite:

```bibtex
@article{yourname2025domain,
  title={Domain-Adaptive Zero-Shot Image Classification via Auto-Tuned CLIP and Large Language Models},
  author={Your Name and Others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025}
}
```
