# 📋 Quick Reference: Figures for Paper

## ✅ Generated Successfully (All 14 Figures)

Location: `s:\Siddu\Final Year\zero-shot\paper\figures\`

---

## 🎯 Essential Figures for Main Paper (Use These 6-7)

| # | Figure | File | Section | Priority |
|---|--------|------|---------|----------|
| 1 | System Architecture | `system_architecture.pdf` | Introduction/Methodology | ⭐⭐⭐ |
| 2 | Performance Table | `performance_table.pdf` | Results (TABLE I) | ⭐⭐⭐ |
| 3 | Ablation Study | `ablation_analysis.pdf` | Results | ⭐⭐ |
| 4 | Adaptation Curves | `adaptation_curves.pdf` | Results | ⭐⭐ |
| 5 | Domain Breakdown | `domain_breakdown_table.pdf` | Results (TABLE II) | ⭐ |
| 6 | Efficiency Analysis | `computational_efficiency.pdf` | Results | ⭐ |
| 7 | Qualitative Examples | `qualitative_examples_template.pdf` | Results | ⭐ (needs real images) |

---

## 📊 Supplementary Material (Move Here)

| # | Figure | File | Purpose |
|---|--------|------|---------|
| S1 | Workflow Diagram | `workflow_diagram.pdf` | Process details |
| S2 | Hyperparameter Sensitivity | `hyperparameter_sensitivity.pdf` | Parameter tuning |
| S3 | Calibration Analysis | `calibration_analysis.pdf` | Confidence quality |
| S4 | Error Analysis | `error_analysis.pdf` | Detailed breakdowns |
| S5 | Statistical Significance | `statistical_significance.pdf` | Statistical tests |
| S6 | Confusion Matrices | `confusion_matrices.pdf` | Per-class errors |
| S7 | Runtime Breakdown | `runtime_breakdown.pdf` | Timing details |

---

## 📝 LaTeX Quick Insert

```latex
% Figure 1: Architecture
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\textwidth]{system_architecture.pdf}
  \caption{System architecture.}
  \label{fig:arch}
\end{figure}

% Table I: Performance
\begin{figure*}[htbp]
  \centering
  \includegraphics[width=\textwidth]{performance_table.pdf}
  \caption{Performance comparison.}
  \label{tab:perf}
\end{figure*}

% Figure 2: Ablation
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\columnwidth]{ablation_analysis.pdf}
  \caption{Ablation study.}
  \label{fig:ablation}
\end{figure}
```

---

## 🔄 Regenerate All Figures

```bash
cd "s:\Siddu\Final Year\zero-shot\paper"
python generate_all_visualizations.py
```

Or individually:
```bash
python create_all_diagrams.py        # Generates figures 1-8
python create_analysis_tables.py     # Generates figures 9-14
```

---

## ⚠️ Important: Update with Real Data

**Current Status:** Using example/placeholder data

**Action Required:**
1. Run experiments: `python run_experiments.py`
2. Edit scripts to load real results
3. Regenerate: `python generate_all_visualizations.py`

---

## 📐 File Sizes & Formats

- **PDF**: Vector graphics (scalable, best for LaTeX)
- **PNG**: Raster graphics (300 DPI, for presentations)
- **Total**: 28 files (14×2 formats)
- **Size**: ~500KB - 2MB per figure

---

## ✨ What Makes These Publication-Quality

✅ 300 DPI resolution  
✅ Times New Roman font (matches IEEE)  
✅ Clear labels and legends  
✅ Color-blind safe palettes  
✅ Grayscale print tested  
✅ Vector graphics (PDF)  
✅ Professional styling  

---

## 📞 Quick Help

- **Full Documentation**: `VISUALIZATIONS_GUIDE.md`
- **Summary**: `FIGURES_SUMMARY.md`
- **Scripts**: `create_all_diagrams.py`, `create_analysis_tables.py`

---

**Status:** ✅ All figures generated and ready to use!
