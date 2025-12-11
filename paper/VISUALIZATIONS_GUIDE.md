# Comprehensive Guide: Diagrams, Tables, and Visualizations

This document explains all the diagrams, tables, and visualizations created for your research paper on zero-shot image classification.

## 📊 Overview

Three main scripts generate all visualizations:
1. **create_all_diagrams.py** - System architecture, workflows, and core visualizations
2. **create_analysis_tables.py** - Detailed analysis tables and statistical charts
3. **generate_figures.py** - Experimental results and performance graphs (from previous session)

## 🎨 Generated Visualizations

### 1. System Architecture Diagram
**File:** `system_architecture.pdf/png`

**Description:** Complete system architecture showing all components and their interactions.

**Components shown:**
- Input Image + Domain Hint
- Domain Inference module
- CLIP Image Encoder (ViT-L/14)
- CLIP Text Encoder (ViT-L/14)
- Domain-Aware Prompt Generation
- Class Prototypes (Adaptive)
- Cosine Similarity Computation
- Top-K Selection
- Caption Generator (BLIP-2)
- LLM Reasoning (Gemini 1.5)
- Confidence Check (threshold > 0.15)
- Adaptive Update (EMA: q ← (1-α)q + αv)
- Final Prediction + Explanation

**Color coding:**
- Blue (#E8F4F8) - Input layer
- Light blue (#B8E6F0) - Processing modules
- Teal (#7FCDDE) - Model components
- Orange (#FFE5B4) - Decision points
- Green (#C8E6C9) - Output layer

**Use in paper:** Introduction or Methodology section to show complete system overview.

---

### 2. Project Workflow Diagram
**File:** `workflow_diagram.pdf/png`

**Description:** Three-phase workflow showing initialization, inference, and adaptive learning.

**Phase 1: Initialization**
1. Load CLIP ViT-L/14
2. Load BLIP-2 Captioner
3. Initialize LLM API
4. Create Class Prototypes

**Phase 2: Inference Pipeline**
1. Input Image + Hint
2. Domain Inference
3. Generate Prompts
4. CLIP Encoding
5. Compute Similarity
6. Caption Generation
7. LLM Reasoning
8. Final Prediction

**Phase 3: Adaptive Learning**
- Confidence Check (> 0.15?)
- Update Prototype (if yes)
- Skip Update (if no)
- Return: Label + Confidence + Explanation

**Use in paper:** Methodology section to explain the complete pipeline.

---

### 3. Performance Comparison Table
**File:** `performance_table.pdf/png`

**Description:** Comprehensive table comparing all methods across datasets.

**Methods compared:**
- CLIP Baseline
- CLIP Ensemble (80 prompts)
- DCLIP (LLM prompts)
- WaffleCLIP
- AutoCLIP
- TPT (Test-time tuning)
- Ours (w/o adaptation)
- Ours (full)
- Ours (+ 500 samples)

**Datasets:**
- ImageNet
- ChestX-ray
- EuroSAT
- Oxford Pets
- Food-101
- Average

**Highlights:**
- Best results highlighted in green
- Our methods highlighted in light green background
- Improvements over baseline shown in rightmost column

**Use in paper:** Results section, main comparison table.

---

### 4. Ablation Analysis
**File:** `ablation_analysis.pdf/png`

**Description:** Two-part visualization showing component contributions and domain-specific improvements.

**Left chart: Component Ablation Study**
- Baseline: 68.3%
- + Domain Prompts: 70.2% (+1.9%)
- + Adaptive Learning: 71.8% (+3.5%)
- + LLM Reasoning: 72.9% (+4.6%)
- + Caption Generation: 73.5% (+5.2%)
- Full System: 74.1% (+5.8%)

**Right chart: Domain-Specific Prompt Impact**
Shows accuracy improvement from generic → domain-aware prompts across:
- Natural: 82.3% → 83.1% (+0.8%)
- Medical: 42.1% → 48.7% (+6.6%)
- Satellite: 51.2% → 58.3% (+7.1%)
- Anime: 65.4% → 68.9% (+3.5%)
- Sketch: 58.7% → 62.3% (+3.6%)

**Use in paper:** Results section, ablation study subsection.

---

### 5. Adaptation Learning Curves
**File:** `adaptation_curves.pdf/png`

**Description:** Six subplots showing how accuracy improves with unlabeled adaptation samples.

**Domains analyzed:**
1. Natural (ImageNet): 68.3% → 73.2% (+4.9%)
2. Medical (ChestX-ray): 42.1% → 52.7% (+10.6%)
3. Satellite (EuroSAT): 51.2% → 63.4% (+12.2%)
4. Anime: 65.4% → 69.8% (+4.4%)
5. Sketch: 58.7% → 63.1% (+4.4%)
6. Average Across All: 57.1% → 64.4% (+7.3%)

**X-axis:** Number of adaptation samples (0, 50, 100, 200, 300, 500)
**Y-axis:** Top-1 Accuracy (%)

**Features:**
- Error bars showing standard deviation
- Shaded region for uncertainty
- Start/end points annotated
- Total gain displayed

**Use in paper:** Results section, showing adaptive learning effectiveness.

---

### 6. Confusion Matrices
**File:** `confusion_matrices.pdf/png`

**Description:** Side-by-side confusion matrices comparing baseline vs. our method.

**Left:** CLIP Baseline (85.0% accuracy)
**Right:** Our Method (98.0% accuracy)

**5 classes shown:**
- Cat
- Dog
- Bird
- Car
- Airplane

**Color coding:**
- Blues colormap for baseline
- Greens colormap for our method
- Darker = higher frequency

**Use in paper:** Results section, qualitative analysis showing error reduction.

---

### 7. Computational Efficiency
**File:** `computational_efficiency.pdf/png`

**Description:** Two-part analysis of latency and memory consumption.

**Left chart: Inference Latency**
- CLIP Baseline: 12.3 ms
- AutoCLIP: 15.7 ms
- TPT (64 aug): 892.4 ms ⚠️ Very slow
- Ours (no LLM): 18.2 ms
- Ours (full): 156.3 ms

**Right chart: GPU Memory**
- CLIP Baseline: 2048 MB
- AutoCLIP: 2048 MB
- TPT: 8192 MB ⚠️ High memory
- Ours (no LLM): 2512 MB
- Ours (full): 3072 MB

**Use in paper:** Results section, efficiency analysis.

---

### 8. Hyperparameter Sensitivity
**File:** `hyperparameter_sensitivity.pdf/png`

**Description:** Four subplots analyzing sensitivity to key hyperparameters.

**Top-left: Learning Rate (α)**
- Range: 0.01 to 0.5
- Optimal: 0.05 (72.5% accuracy)
- Shows inverted U-shape

**Top-right: Confidence Threshold (τ)**
- Range: 0.05 to 0.5
- Optimal: 0.15 (72.5% accuracy)
- Balance between quality and quantity

**Bottom-left: Number of Prompts**
- Dual Y-axis: Accuracy vs. Latency
- Range: 1 to 200 prompts
- Recommended: 100 (good accuracy/speed tradeoff)

**Bottom-right: Text-Visual Weight Ratio**
- Optimal: 0.7 text / 0.3 visual (74.1% accuracy)
- Shows importance of text guidance

**Use in paper:** Results section or Appendix, hyperparameter tuning details.

---

### 9. Domain Breakdown Table
**File:** `domain_breakdown_table.pdf/png`

**Description:** Detailed metrics for each domain with our full method.

**Metrics shown:**
- Top-1 Accuracy
- Top-5 Accuracy
- ECE (Expected Calibration Error)
- Average Confidence
- Inference Time (ms)
- Number of Samples

**Domains:**
1. Natural Images (ImageNet): 74.1% / 91.2%
2. Medical (ChestX-ray): 52.7% / 78.9%
3. Satellite (EuroSAT): 63.4% / 85.3%
4. Anime (Danbooru): 69.8% / 88.5%
5. Sketches (Sketchy): 63.1% / 83.7%
6. Average: 64.6% / 85.5%

**Use in paper:** Results section, detailed breakdown table.

---

### 10. Statistical Significance Table
**File:** `statistical_significance.pdf/png`

**Description:** Statistical test results comparing all methods to ours.

**Metrics:**
- Accuracy (mean ± std)
- p-value (from paired t-test)
- Significance stars (*** / ** / *)
- Verbal significance level

**All comparisons:**
- vs. Our Full Method (reference)
- Most improvements are highly significant (p < 0.001)
- Shows statistical robustness

**Use in paper:** Results section or Appendix, statistical validation.

---

### 11. Calibration Analysis
**File:** `calibration_analysis.pdf/png`

**Description:** Six reliability diagrams showing calibration quality.

**Methods analyzed:**
1. CLIP Baseline (ECE = 0.145) - Fair
2. AutoCLIP (ECE = 0.098) - Fair
3. WaffleCLIP (ECE = 0.112) - Fair
4. Ours w/o LLM (ECE = 0.072) - Good
5. Ours w/o adapt (ECE = 0.063) - Good
6. Ours full (ECE = 0.047) - Excellent

**Features:**
- Perfect calibration line (diagonal)
- Actual accuracy bars
- Sample counts per bin
- Red/green shading for over/under-confidence

**Use in paper:** Results section, showing confidence calibration quality.

---

### 12. Error Analysis
**File:** `error_analysis.pdf/png`

**Description:** Four-part comprehensive error analysis.

**Top-left: Error Type Distribution**
- Fine-grained Confusion: 45 → 25 (44% reduction)
- Cross-domain Error: 25 → 10 (60% reduction)
- Ambiguous Sample: 20 → 8 (60% reduction)
- True Misclassification: 10 → 5 (50% reduction)

**Top-right: Top-K Accuracy Progression**
- Shows baseline vs. ours for K=1,2,3,5,10
- Improvement at each K level

**Bottom-left: Per-Class Accuracy Distribution**
- Histogram of 100 classes
- Shows shift toward higher accuracy

**Bottom-right: Confidence vs. Accuracy Scatter**
- 500 random samples
- Shows better calibration for our method

**Use in paper:** Results section, detailed error analysis.

---

### 13. Runtime Breakdown
**File:** `runtime_breakdown.pdf/png`

**Description:** Detailed timing analysis.

**Left: Pie Chart**
Component breakdown:
- CLIP Encoding: 18.2 ms (11.6%)
- Caption Generation: 92.3 ms (59.0%)
- LLM Reasoning: 41.5 ms (26.5%)
- Similarity Computation: 2.1 ms (1.3%)
- Prototype Update: 1.8 ms (1.2%)
- Other: 0.4 ms (0.3%)

**Right: Configuration Table**
Shows latency and throughput for:
- CLIP Only: 12.3 ms / 81.3 img/s
- + Domain Prompts: 15.7 ms / 63.7 img/s
- + Caption: 104.5 ms / 9.6 img/s
- + Adaptive: 18.2 ms / 54.9 img/s
- Full: 156.3 ms / 6.4 img/s
- Full (cached LLM)*: 64.1 ms / 15.6 img/s

**Use in paper:** Results section, efficiency analysis.

---

### 14. Qualitative Examples Template
**File:** `qualitative_examples_template.pdf/png`

**Description:** 3×4 grid layout for showing example predictions.

**Structure:**
- Row 1: Success cases (Natural, Medical, Satellite, Anime)
- Row 2: Failure cases (Natural, Medical, Satellite, Sketch)
- Row 3: Challenging cases (4 difficult examples)

**Each cell shows:**
- [Placeholder for actual image]
- Ground Truth label (green box)
- Baseline prediction (red box)
- Our prediction (blue box)

**Instructions:**
1. Replace placeholder with actual images
2. Fill in actual predictions
3. Use for qualitative analysis section

**Use in paper:** Results section, qualitative examples.

---

## 🚀 How to Generate All Figures

### Option 1: Run Master Script (Recommended)
```bash
cd paper
python generate_all_visualizations.py
```

This runs all three visualization scripts automatically.

### Option 2: Run Individual Scripts
```bash
cd paper
python create_all_diagrams.py
python create_analysis_tables.py
python generate_figures.py  # If available
```

### Option 3: Generate from Python
```python
import subprocess
subprocess.run(['python', 'generate_all_visualizations.py'])
```

## 📁 Output Structure

All figures are saved to `paper/figures/` in both PDF and PNG formats:

```
figures/
├── system_architecture.pdf
├── system_architecture.png
├── workflow_diagram.pdf
├── workflow_diagram.png
├── performance_table.pdf
├── performance_table.png
├── ablation_analysis.pdf
├── ablation_analysis.png
├── adaptation_curves.pdf
├── adaptation_curves.png
├── confusion_matrices.pdf
├── confusion_matrices.png
├── computational_efficiency.pdf
├── computational_efficiency.png
├── hyperparameter_sensitivity.pdf
├── hyperparameter_sensitivity.png
├── domain_breakdown_table.pdf
├── domain_breakdown_table.png
├── statistical_significance.pdf
├── statistical_significance.png
├── calibration_analysis.pdf
├── calibration_analysis.png
├── error_analysis.pdf
├── error_analysis.png
├── runtime_breakdown.pdf
├── runtime_breakdown.png
└── qualitative_examples_template.pdf
└── qualitative_examples_template.png
```

## 📝 Using Figures in LaTeX

### For main paper (research_paper.tex)

Add to preamble:
```latex
\usepackage{graphicx}
\graphicspath{{figures/}}
```

Insert figures:
```latex
% System architecture
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\textwidth]{system_architecture.pdf}
  \caption{Complete system architecture showing all components and data flow.}
  \label{fig:architecture}
\end{figure}

% Performance table
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{performance_table.pdf}
  \caption{Performance comparison across methods and datasets.}
  \label{fig:performance}
\end{figure}

% Ablation study
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{ablation_analysis.pdf}
  \caption{Ablation study showing component contributions and domain-specific improvements.}
  \label{fig:ablation}
\end{figure}
```

### For two-column IEEE format:
```latex
% Use \columnwidth for single column
\includegraphics[width=\columnwidth]{figure.pdf}

% Use \textwidth for full-width (both columns)
\begin{figure*}[htbp]
  \centering
  \includegraphics[width=\textwidth]{wide_figure.pdf}
  \caption{Wide figure spanning both columns.}
  \label{fig:wide}
\end{figure*}
```

## 🎨 Customization

### Change Colors
Edit the scripts and modify color definitions:

```python
# In create_all_diagrams.py
input_color = '#E8F4F8'  # Change to your preferred color
process_color = '#B8E6F0'
model_color = '#7FCDDE'
# ... etc
```

### Adjust Sizes
```python
# Change figure size
fig, ax = plt.subplots(figsize=(14, 10))  # width, height in inches
```

### Modify Fonts
```python
plt.rcParams['font.size'] = 10  # Base font size
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
```

### Change DPI
```python
plt.rcParams['figure.dpi'] = 300  # For screen display
plt.rcParams['savefig.dpi'] = 300  # For saving files
```

## 📊 Data Sources

### Current Status: Example Data
All visualizations currently use **example/placeholder data** for demonstration purposes.

### Next Steps: Add Real Data

1. **Run Experiments**
   ```bash
   cd ..
   python run_experiments.py --output results/
   ```

2. **Update Scripts with Real Data**
   - Edit `create_all_diagrams.py`
   - Replace example arrays with actual results from `results/*.json`
   
3. **Regenerate Figures**
   ```bash
   python generate_all_visualizations.py
   ```

## 🔍 Quality Checklist

Before including in paper:

- [ ] All figures use PDF format (vector graphics)
- [ ] Font sizes are readable (minimum 8pt)
- [ ] Colors are distinguishable when printed in grayscale
- [ ] Axis labels are clear and have units
- [ ] Legends are positioned appropriately
- [ ] Captions are descriptive and self-contained
- [ ] All figures are referenced in the text
- [ ] Figure numbers match references
- [ ] Statistical significance is clearly marked
- [ ] Error bars/confidence intervals are shown
- [ ] Source code and data are documented

## 📚 Recommended Paper Structure

### Main Paper Figures:
1. **Figure 1:** System Architecture → Introduction
2. **Figure 2:** Workflow Diagram → Methodology
3. **Table I:** Performance Comparison → Results
4. **Figure 3:** Ablation Analysis → Results
5. **Figure 4:** Adaptation Curves → Results
6. **Figure 5:** Computational Efficiency → Results
7. **Figure 6:** Qualitative Examples → Results

### Supplementary Material:
8. **Figure S1:** Hyperparameter Sensitivity
9. **Figure S2:** Calibration Analysis
10. **Figure S3:** Error Analysis
11. **Figure S4:** Runtime Breakdown
12. **Table SI:** Domain Breakdown
13. **Table SII:** Statistical Significance
14. **Figure S5:** Confusion Matrices

## 🛠️ Troubleshooting

### Issue: "No module named matplotlib"
```bash
pip install matplotlib numpy seaborn scipy scikit-learn
```

### Issue: Fonts not rendering correctly
```python
# Use system default fonts
plt.rcParams['font.family'] = 'sans-serif'
```

### Issue: Figures too large for paper
```python
# Reduce figure size
fig, ax = plt.subplots(figsize=(8, 6))  # Smaller dimensions
```

### Issue: Text overlapping in plots
```python
plt.tight_layout()  # Add before saving
# or
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
```

## 📖 Additional Resources

- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [IEEE Graphics Guidelines](https://www.ieee.org/publications/authors/author-graphics-guidelines.html)
- [Nature Figure Guidelines](https://www.nature.com/nature/for-authors/final-submission)

## ✨ Tips for Publication-Quality Figures

1. **Use vector graphics (PDF)** for line plots, diagrams
2. **Use high-res raster (PNG 300 DPI)** for photos, heatmaps
3. **Keep it simple** - remove unnecessary decorations
4. **Consistent styling** - same fonts, colors across all figures
5. **Readable legends** - place where they don't obscure data
6. **Color-blind friendly** - use patterns in addition to colors
7. **Test print** - check how it looks in grayscale
8. **Self-contained captions** - readers should understand without reading text

## 🎯 Final Checklist

Before submission:
- [ ] All figures generated successfully
- [ ] Real experimental data used (not example data)
- [ ] Figures referenced in paper text
- [ ] Captions written and descriptive
- [ ] LaTeX compiles without errors
- [ ] PDF output shows all figures correctly
- [ ] File sizes reasonable (< 1MB per figure)
- [ ] No copyright issues (all content original)
- [ ] Figures match journal format requirements
- [ ] Supplementary materials prepared

---

**Good luck with your paper submission! 🚀**
