# Chemical Diversity Selection & Validation Toolkit

A Python-based workflow designed to **select structurally diverse molecular subsets** from large chemical libraries and **validate chemical space coverage** with **publication-quality plots**.

This toolkit uses **RDKit** for parallelized **Butina clustering** on **ECFP4 (Morgan) fingerprints** with **Tanimoto similarity**, then validates the chosen subset via **physicochemical properties**, **Murcko scaffolds**, **PAINS alerts**, and **chemical space mapping** (e.g., PCA).

---

## Description

Reducing huge chemical libraries into manageable, representative subsets is a common bottleneck for:
- virtual screening pipelines,
- docking/MD campaigns,
- experimental purchasing/testing,
- broad scaffold exploration and chemical space coverage analysis.

This repository solves that in **two stages**:

### 1) Selection (Diversity Picking)
- Generates Morgan fingerprints (ECFP4)
- Computes pairwise distances efficiently (parallelized)
- Runs **Butina clustering**
- Selects **representative cluster centroids** to match a target subset size

### 2) Validation (Chemical Space Coverage)
Compares the **subset vs the full library** using:
- physicochemical property distributions
- **Murcko scaffold** coverage
- **PAINS** (Pan-Assay Interference Compounds) alerts
- chemical space mapping (**PCA**)
- overlap plots (radar), similarity/coverage plots (heatmaps, histograms), and distribution plots (KDE)

---

## Key Features

- **Parallelized Processing**
  - Fast fingerprint generation and distance calculations using `multiprocessing`.

- **Butina Clustering (ECFP4/Tanimoto)**
  - Selects cluster representatives based on a user-defined **distance cutoff**.

- **Structural Validation**
  - Automated **Murcko Scaffolds** analysis and **PAINS** alerts.

- **Automated Reporting**
  - Generates a structured **TXT** statistical report.
  - Writes **SDF** outputs with cluster metadata.

- **Publication-Ready Plots**
  - PCA visualizations
  - Radar plots for property overlap
  - Similarity heatmaps
  - Coverage histograms and distribution plots (KDE)

---

## Understanding Butina Cutoff Thresholds

The selector uses a **Distance Threshold**:

\[
Distance = 1 - Similarity
\]

Choosing the cutoff is critical because it defines the “neighborhood radius” of a cluster:

- **Cutoff 0.20** *(Sim ≥ 0.80)* → Very tight clusters (close analogs)
- **Cutoff 0.40** *(Sim ≥ 0.60)* → Moderate similarity
- **Cutoff 0.65** *(Sim ≥ 0.35)* → Diverse clustering (**recommended for broad scaffold hopping**)
- **Cutoff 0.80** *(Sim ≥ 0.20)* → Very loose clustering

**Default:** `0.65` to ensure high structural diversity and broad chemical space coverage.

---

## Requirements

- **Python:** `>= 3.8`

### Dependencies
- `rdkit`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## Installation

### Option 1: pip
```bash
pip install numpy pandas matplotlib seaborn scikit-learn rdkit
```
---

## Citation

If you use this repository/scripts in academic work, please cite:

Isaoğlu, M., & Durdağı, S. (2025). *Butina Diversity Selection & Subset Validation Tool* (Version 1.0) [Source code]. https://github.com/DurdagiLab/butina-diversity-toolkit
