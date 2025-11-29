# Figure Generation Notes

**Date:** November 29, 2025  
**Purpose:** Publication-ready figures for IEEE conference paper

---

## Changes Made

### 1. **Moved to Paper Folder**
- All figures now saved directly to `/paper/` directory
- Ready for LaTeX inclusion without path issues
- Old `analysis/figures/` directory removed to avoid confusion

### 2. **Removed Titles from Plots**
- Titles now provided by LaTeX captions
- Cleaner appearance for publication
- Follows IEEE conference figure guidelines

### 3. **Increased Legibility**
- **Font sizes increased:**
  - Base font: 10 → 14
  - Axis labels: 11 → 16
  - Tick labels: 9 → 13
  - Legend: 9 → 12
- **Better for:**
  - Print readability
  - Projection on screens
  - Color-blind accessibility

### 4. **Optimized Figure Size**
- Changed from (10, 6) to (8, 5) inches
- Better aspect ratio for IEEE single-column figures
- Will scale well when rendered at 0.48\textwidth

---

## Generated Figures

### **For Paper (4 figures included in manuscript):**

#### 1. **tolerance_comparison.png** (236 KB)
- **Purpose:** Show precision impact is negligible
- **Content:** 3 bar charts comparing scale 7 vs 10
- **Key feature:** 1.002× ratio annotation
- **LaTeX reference:** `\ref{fig:tolerance_comparison}`

#### 2. **layer_comparison.png** (230 KB)
- **Purpose:** Rank layers by proof generation time
- **Content:** Horizontal bar chart, 8 layers
- **Key feature:** Color gradient showing 146× range
- **LaTeX reference:** `\ref{fig:layer_comparison}`

#### 3. **scaling_curves.png** (296 KB)
- **Purpose:** Demonstrate sub-linear scaling
- **Content:** Log-log scatter plot with power-law fit
- **Key feature:** 0.65 exponent annotation
- **LaTeX reference:** `\ref{fig:scaling_curves}`

#### 4. **composite_comparison.png** (195 KB)
- **Purpose:** Compare composite architectures
- **Content:** Two bar charts (proof time, circuit size)
- **Key feature:** CNN-ReLU shown as OOM with hatching
- **LaTeX reference:** `\ref{fig:composite_comparison}`

### **Additional Figures (not included, available if needed):**

#### 5. **performance_tiers.png** (175 KB)
- **Purpose:** Show Fast/Medium/Slow distribution
- **Content:** Bar chart + box plots
- **Redundant with:** Table III already shows tiers

#### 6. **memory_analysis.png** (283 KB)
- **Purpose:** Memory usage patterns
- **Content:** Scatter plot + efficiency bars
- **Notes:** Interesting but not critical for main narrative

#### 7. **system_architecture_diagram.png** (382 KB)
- **Purpose:** Overview of benchmarking pipeline
- **Content:** Workflow diagram (PyTorch → ONNX → EZKL → Proof)
- **Status:** Already existed, referenced in methodology

---

## LaTeX Integration

### **Current References in Paper:**

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.48\textwidth]{tolerance_comparison.png}
    \caption{...}
    \label{fig:tolerance_comparison}
\end{figure}
```

### **All Figures Referenced:**
- ✅ Figure 1: tolerance_comparison.png (after Table II)
- ✅ Figure 2: layer_comparison.png (after Table III)
- ✅ Figure 3: scaling_curves.png (after Table IV)
- ✅ Figure 4: composite_comparison.png (after Table V)
- ✅ Figure (methodology): system_architecture_diagram.png

---

## Style Guidelines Applied

### **IEEE Conference Standards:**
- High resolution (300 DPI)
- Serif fonts for better print quality
- Clear axis labels with units
- Legible at 50% scale
- Grayscale-friendly color schemes
- Professional appearance

### **Accessibility:**
- Large, readable fonts (13-16pt)
- High contrast colors
- Clear markers in scatter plots
- Patterns (hatching) for failed experiments
- Multiple visual cues (not color-only)

### **Scientific Clarity:**
- No decorative elements
- Grid lines for value reading
- Annotations for key findings
- Consistent color schemes across figures
- Direct value labels where helpful

---

## Regeneration Instructions

If figures need to be regenerated:

```bash
cd /home/s15/CascadeProjects/cnn_ezkl_bench
python plot_results.py
```

**Script automatically:**
- Loads all 26 experiment results
- Generates 6 figures with updated styles
- Saves to `paper/` directory
- Reports completion status

**To customize:**
- Edit `plot_results.py`
- Adjust font sizes in lines 19-24
- Modify figure sizes in line 25
- Change output directory in function definitions

---

## File Sizes

| Figure | Size | Format | DPI |
|--------|------|--------|-----|
| tolerance_comparison.png | 236 KB | PNG | 300 |
| layer_comparison.png | 230 KB | PNG | 300 |
| scaling_curves.png | 296 KB | PNG | 300 |
| composite_comparison.png | 195 KB | PNG | 300 |
| performance_tiers.png | 175 KB | PNG | 300 |
| memory_analysis.png | 283 KB | PNG | 300 |
| system_architecture_diagram.png | 382 KB | PNG | 300 |

**Total:** ~1.7 MB for all figures (well within limits)

---

## Version Control

### **Old Location (removed):**
- `analysis/figures/*.png` - Generated Nov 29, 11:00 PM
- Had titles in plots
- Smaller fonts
- Not optimized for publication

### **New Location (current):**
- `paper/*.png` - Generated Nov 29, 11:49 PM
- No titles (use LaTeX captions)
- Larger, more legible fonts
- Publication-ready

---

## Quality Checklist

### **Visual Quality:**
- ✅ High resolution (300 DPI)
- ✅ Clear axis labels
- ✅ Readable at 50% scale
- ✅ Professional appearance
- ✅ Consistent styling

### **Scientific Accuracy:**
- ✅ All data from actual experiments
- ✅ Annotations match analysis
- ✅ Units clearly labeled
- ✅ Legends complete
- ✅ No misleading visual tricks

### **Accessibility:**
- ✅ Large fonts (13-16pt)
- ✅ High contrast
- ✅ Patterns for categories
- ✅ Not color-dependent
- ✅ Clear labels

### **Publication Standards:**
- ✅ Follows IEEE guidelines
- ✅ Captions in LaTeX (not on figures)
- ✅ Referenced in text
- ✅ Proper numbering
- ✅ High quality for print

---

## Notes on Specific Figures

### **tolerance_comparison.png:**
- Shows THE most important finding
- Three panels side-by-side
- Clear 1.002× ratio annotation
- "Expected vs Actual" annotation on left panel
- **Keep this prominent in paper**

### **layer_comparison.png:**
- Horizontal bars easier to read than vertical
- Color gradient from green (fast) to red (slow)
- Constraint counts shown for context
- Alphabetically ordered by speed
- **Very practical for practitioners**

### **scaling_curves.png:**
- Log-log plot shows power-law relationship
- Different markers for different layer types
- Dashed line shows fit with equation
- Yellow box highlights sub-linear exponent
- **Critical for scalability claims**

### **composite_comparison.png:**
- Two panels: time and circuit size
- CNN-ReLU shown with hatching (infeasible)
- Direct comparison of working architectures
- Shows 4.2× speedup of CNN-Strided
- **Validates design recommendations**

---

## Future Improvements (if needed)

### **Potential Enhancements:**
- Add error bars (currently std dev in tables)
- Color-code by deployment regime in layer chart
- Add Pareto frontier overlay on scaling plot
- Interactive versions for online supplementary materials

### **Alternative Formats:**
- EPS/PDF for vector graphics (larger file size)
- SVG for web version
- TIFF for high-quality print

### **Additional Figures:**
- Pareto frontier (time vs size vs constraints)
- Memory breakdown by operation
- GPU vs RAM utilization comparison
- Cost-per-operation comparison

---

## Conclusion

All figures are **publication-ready** and located in the `paper/` directory. They follow IEEE conference standards, have excellent legibility, and directly support the paper's key findings.

**No further action needed unless revisions requested by reviewers.**
