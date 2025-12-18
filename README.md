# Chemical Diversity Workflow (RDKit)
**Butina-based diverse subset selection + comprehensive subset validation**

This repository provides an end-to-end RDKit workflow to:
1) Select a structurally diverse subset from a parent SDF library using **Butina clustering** (ECFP4/Morgan radius=2, 2048-bit; Tanimoto distance). :contentReference[oaicite:0]{index=0}  
2) Validate the selected subset vs. the full library using descriptors, scaffold diversity, PAINS screening, coverage analysis, and publication-style plots. :contentReference[oaicite:1]{index=1}  

---

## Scripts

### 1) `butina_diversity_selector.py`  (Selection)
Selects representative molecules (cluster centroids) using Butina clustering and exports:
- Selected subset **SDF**
- A concise **TXT report**
- PCA chemical space plot + cluster-size distribution plot :contentReference[oaicite:2]{index=2}  

**Main settings (edit in `__main__`):**
- `INPUT_SDF`, `OUTPUT_SDF`
- `TARGET_COUNT`
- `CUTOFF_DISTANCE` (distance = 1 − similarity)
- `CPU_CORES` :contentReference[oaicite:3]{index=3}  

---

### 2) `diversity_subset_validation.py` (Validation)
Validates the selected subset against the parent library and exports:
- Detailed **TXT report**
- Distribution plots (MW, LogP, TPSA, HBD, HBA, RotB)
- Radar plot, similarity heatmap, and coverage plot :contentReference[oaicite:4]{index=4}  

**Main settings (edit at top of the file):**
- `FULL_LIB_PATH` (parent SDF)
- `SUBSET_LIB_PATH` (selected subset SDF)
- `REPORT_FILE`
- `PLOT_PREFIX` :contentReference[oaicite:5]{index=5}  

---

## Installation

Recommended (conda):
```bash
conda create -n chemdiv python=3.10 -y
conda activate chemdiv
conda install -c conda-forge rdkit -y
pip install numpy pandas matplotlib seaborn scikit-learn

## Citation
If you use this repository/scripts in academic work, please cite:

Isaoğlu, M., & Durdağı, S. (2025). *Butina Diversity Selection & Subset Validation Tool* (Version 1.0) [Source code]. https://github.com/DurdagiLab/butina-diversity-toolkit
