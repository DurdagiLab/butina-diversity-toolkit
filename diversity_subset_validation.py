"""
#################################################################################################################################
Title: Diversity Subset Validation (RDKit) - Descriptors, Scaffolds, PAINS, Coverage & Plots

Developed by: Mine Isaoglu, Ph.D.
Principal Investigator: Serdar Durdagi, Ph.D.
Affiliation: Computational Drug Design Center (HITMER),
             Faculty of Pharmacy, Bahçeşehir University, Istanbul, Turkey

Version: 2025-12 (December 2025)

Description:
    End-to-end validation of a selected chemical subset against a parent library.
    Computes physicochemical descriptors, Murcko scaffold diversity, PAINS alerts,
    coverage via ECFP4/Tanimoto distances, and generates publication-style figures
    plus a structured TXT report.

Inputs:
    - FULL_LIB_PATH: parent library SDF
    - SUBSET_LIB_PATH: selected subset SDF

Outputs:
    - REPORT_FILE: TXT report
    - PNG figures with prefix PLOT_PREFIX

#################################################################################################################################
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import FilterCatalog

# ==========================================
# SETTINGS
# ==========================================
FULL_LIB_PATH = "FILE_NAME.sdf"  		       # Original library file
SUBSET_LIB_PATH = "FILE_NAME_final_selection.sdf"      # Selected 500-compound subset file
REPORT_FILE = "validation_report_FILE_NAME.txt"        # Report output file
PLOT_PREFIX = "FILE_NAME"                              # Plot filename prefix
# ==========================================

# --- Style Settings ---
sns.set_style("white") 
sns.set_context("paper", font_scale=1.2)

def get_extended_props(mol):
    if mol is None: return None
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotB": Lipinski.NumRotatableBonds(mol)
    }

def get_fp(mol):
    if mol is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def get_scaffold(mol):
    if mol is None: return None
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return s if s else "Acyclic"
    except:
        return None

def check_pains(mols):
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    pains_found = []
    for i, mol in enumerate(mols):
        if mol is None: continue
        if catalog.HasMatch(mol):
            entry = catalog.GetFirstMatch(mol)
            pains_found.append((i, entry.GetDescription()))
    return pains_found

def save_single_plot(df_full, df_sub, feature, title, xlabel):
    """Single distribution plot (no grid)."""
    plt.figure(figsize=(8, 6))
    sns.despine() 
    
    sns.kdeplot(data=df_full, x=feature, fill=True, color="gray", alpha=0.3, label="Full Library")
    sns.kdeplot(data=df_sub, x=feature, color="red", linewidth=3, label="Selected Set")
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    
    filename = f"{PLOT_PREFIX}_{feature}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"   -> Saved: {filename}")

def plot_radar_separate(df_full, df_sub):
    """Radar plot (dark grid and visible tick labels)."""
    categories = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotB']
    
    df_comb = pd.concat([df_full[categories], df_sub[categories]])
    normalized_full = (df_full[categories] - df_comb.min()) / (df_comb.max() - df_comb.min())
    normalized_sub = (df_sub[categories] - df_comb.min()) / (df_comb.max() - df_comb.min())
    
    mean_full = normalized_full.mean().values.flatten().tolist()
    mean_sub = normalized_sub.mean().values.flatten().tolist()
    mean_full += mean_full[:1]
    mean_sub += mean_sub[:1]
    
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, color='black', size=12)
    ax.set_rlabel_position(0)
    
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="black", size=9, fontweight='bold')
    plt.ylim(0, 1)
    
    ax.grid(color='#555555', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.spines['polar'].set_visible(False) 

    ax.plot(angles, mean_full, linewidth=1, linestyle='solid', label="Full Library", color="gray")
    ax.fill(angles, mean_full, 'gray', alpha=0.1)
    
    ax.plot(angles, mean_sub, linewidth=3, linestyle='solid', label="Selected Set", color="red")
    ax.fill(angles, mean_sub, 'red', alpha=0.25)
    
    plt.title("Overall Property Overlap", y=1.08, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    filename = f"{PLOT_PREFIX}_Radar.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"   -> Saved: {filename}")

def plot_heatmap_separate(mols_sub):
    """Heatmap plot (blue-red)."""
    print("   -> Computing heatmap...")
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols_sub]
    n_sel = len(fps)
    sim_matrix = np.zeros((n_sel, n_sel))
    
    for i in range(n_sel):
        for j in range(i, n_sel):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap="coolwarm", vmin=0, vmax=1, 
                cbar_kws={'label': 'Tanimoto Similarity'})
    
    plt.title("Similarity Matrix of the Selected Molecules", fontsize=14, fontweight='bold')
    plt.xlabel("Molecule ID", fontsize=12)
    plt.ylabel("Molecule ID", fontsize=12)
    
    filename = f"{PLOT_PREFIX}_Heatmap.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"   -> Saved: {filename}")

def run_visual_validation():
    print("--- COMPREHENSIVE VALIDATION AND REPORTING ---")
    start_time = time.time()

    # 1. Read Files
    print(f"[1/6] Reading input files...")
    suppl_full = Chem.SDMolSupplier(FULL_LIB_PATH)
    suppl_sub = Chem.SDMolSupplier(SUBSET_LIB_PATH)
    mols_full = [m for m in suppl_full if m is not None]
    mols_sub = [m for m in suppl_sub if m is not None]
    n_full = len(mols_full)
    n_sub = len(mols_sub)

    # 2. Physicochemical Properties
    print(f"[2/6] Computing physicochemical properties...")
    props_full = [get_extended_props(m) for m in mols_full]
    props_sub = [get_extended_props(m) for m in mols_sub]
    df_full = pd.DataFrame(props_full)
    df_sub = pd.DataFrame(props_sub)

    # 3. Scaffold and PAINS Analysis (Re-included for reporting)
    print(f"[3/6] Performing scaffold and PAINS analysis...")
    scaffolds = [get_scaffold(m) for m in mols_sub]
    unique_scaffolds = set(scaffolds)
    n_unique_scaffold = len(unique_scaffolds)
    scaffold_percent = (n_unique_scaffold / n_sub) * 100
    
    pains_results = check_pains(mols_sub)
    n_pains = len(pains_results)

    # 4. Coverage Analysis Data
    print(f"[4/6] Computing coverage distances...")
    fps_full = [get_fp(m) for m in mols_full]
    fps_sub = [get_fp(m) for m in mols_sub]
    min_distances = []
    for fp_query in fps_full:
        if fp_query is None: continue
        sims = DataStructs.BulkTanimotoSimilarity(fp_query, fps_sub)
        min_distances.append(1.0 - max(sims))
    avg_dist = np.mean(min_distances)
    max_dist_val = max(min_distances)

    # 5. Write Report (DETAILED FORMAT)
    print(f"[5/6] Writing report file: {REPORT_FILE}")
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("        DIVERSITY SUBSET VALIDATION REPORT         \n")
        f.write("==================================================\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("1. GENERAL INFORMATION\n")
        f.write(f"   - Full Library      : {FULL_LIB_PATH} ({n_full} molecules)\n")
        f.write(f"   - Selected Subset   : {SUBSET_LIB_PATH} ({n_sub} molecules)\n")
        f.write(f"   - Selection Ratio   : {(n_sub/n_full)*100:.2f}%\n\n")
        
        f.write("2. STRUCTURAL DIVERSITY (SCAFFOLD DIVERSITY)\n")
        f.write(f"   - Number of Unique Scaffolds : {n_unique_scaffold}\n")
        f.write(f"   - Scaffold Diversity Ratio   : {scaffold_percent:.2f}%\n")
        if scaffold_percent > 90:
            f.write("   - Interpretation             : EXCELLENT (Very high uniqueness)\n\n")
        elif scaffold_percent > 50:
            f.write("   - Interpretation             : GOOD (High diversity)\n\n")
        else:
            f.write("   - Interpretation             : LOW/MODERATE DIVERSITY\n\n")

        f.write("3. COVERAGE ANALYSIS\n")
        f.write(f"   - Mean Representation Distance : {avg_dist:.3f}\n")
        f.write(f"   - Maximum Representation Distance: {max_dist_val:.3f}\n")
        f.write("   (Note: 0.0=Identical, 1.0=Completely different. For highly diverse sets, an average distance of ~0.5–0.6 can be typical.)\n\n")

        f.write("4. PAINS (FALSE POSITIVE) ANALYSIS\n")
        f.write(f"   - Molecules with PAINS Risk    : {n_pains}\n")
        if n_pains > 0:
            f.write("   - Detected Entries (First 3 Examples):\n")
            for idx, desc in pains_results[:3]:
                f.write(f"     * ID {idx}: {desc}\n")
        else:
            f.write("   - Outcome                      : CLEAN (No PAINS detected)\n\n")

        f.write("5. MEAN PHYSICOCHEMICAL PROPERTIES\n")
        f.write(f"   {'Property':<10} | {'Full Library':<15} | {'Selected Set':<15}\n")
        f.write(f"   {'-'*10}-|-{'-'*15}-|-{'-'*15}\n")
        cols = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotB']
        for c in cols:
            f.write(f"   {c:<10} | {df_full[c].mean():<15.2f} | {df_sub[c].mean():<15.2f}\n")
    
    # 6. Figures
    print(f"[6/6] Generating plots...")
    
    # A) Distribution Plots
    feature_map = [
        ('MW', 'Molecular Weight Distribution', 'Molecular Weight (g/mol)'),
        ('LogP', 'LogP (Hydrophobicity) Distribution', 'LogP (Octanol/Water)'),
        ('TPSA', 'TPSA (Polar Surface Area) Distribution', 'TPSA (Å²)'),
        ('HBD', 'Hydrogen Bond Donors (HBD)', 'HBD Count'),
        ('HBA', 'Hydrogen Bond Acceptors (HBA)', 'HBA Count'),
        ('RotB', 'Rotatable Bond Count', 'Rotatable Bond Count')
    ]
    for feat, title, xlabel in feature_map:
        save_single_plot(df_full, df_sub, feat, title, xlabel)

    # B) Radar
    plot_radar_separate(df_full, df_sub)

    # C) Heatmap
    plot_heatmap_separate(mols_sub)

    # D) Coverage Plot
    plt.figure(figsize=(10, 7))
    sns.despine()
    sns.histplot(min_distances, bins=30, color="#2ecc71", kde=True, element="step", label="Molecule Count")
    plt.axvline(avg_dist, color='black', linestyle='--', linewidth=2.5, 
                label=f'Mean Distance: {avg_dist:.2f}')
    plt.title("Coverage Analysis", fontsize=14, fontweight='bold')
    plt.xlabel("Distance to Nearest Representative (Tanimoto Distance)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend(loc='upper left', frameon=True, fontsize=12)
    
    filename_cov = f"{PLOT_PREFIX}_Coverage.png"
    plt.tight_layout()
    plt.savefig(filename_cov, dpi=300)
    plt.close()
    print(f"   -> Saved: {filename_cov}")

    print("-" * 40)
    print(f"PROCESS COMPLETED. Report: {REPORT_FILE}")

if __name__ == "__main__":
    run_visual_validation()