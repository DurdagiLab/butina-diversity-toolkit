"""
################################################################################################################################
Title: Butina Diversity Selector (RDKit) - Parallel ECFP4/Tanimoto Clustering

Developed by: Mine Isaoglu, Ph.D.
Principal Investigator: Serdar Durdagi, Ph.D.
Affiliation: Computational Drug Design Center (HITMER),
             Faculty of Pharmacy, Bahçeşehir University, Istanbul, Turkey

Version: 2025-12 (December 2025)

Description:
    Parallelized workflow to select a structurally diverse subset from an input SDF
    library using Butina clustering on Tanimoto distance computed from Morgan
    fingerprints (radius=2, 2048 bits; ECFP4-equivalent).

Workflow:
    1) Load and validate molecules from SDF
    2) Generate fingerprints in parallel
    3) Compute condensed distance matrix and perform Butina clustering
    4) Select cluster centroids up to a target count
    5) Write selected subset to output SDF with metadata
    6) Export a statistics report (TXT)
    7) Visualize chemical space via PCA and cluster-size distribution

Configuration:
    Parameters are defined in the __main__ block (manual entry).
    (Optional) You may refactor to argparse CLI for reproducible runs.

Notes:
    - Butina cutoff is a DISTANCE threshold where distance = 1 - Tanimoto similarity.
      Example: cutoff=0.65 corresponds to similarity=0.35.
    - RDKit is required.

################################################################################################################################
"""

import sys
import time
import logging
import multiprocessing
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- WORKER FUNCTIONS (must be top-level for multiprocessing) ---

def _generate_fp_worker(mol: Chem.Mol) -> Optional[DataStructs.ExplicitBitVect]:
    """Worker function to generate a Morgan fingerprint (ECFP4-equivalent)."""
    if mol is None:
        return None
    try:
        # Radius 2 ~= ECFP4; 2048 bits is a standard setting for diversity selection
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return None


def _calc_dist_worker(args: Tuple) -> List[float]:
    """Worker function to compute a single row of the condensed distance matrix."""
    i, fp_i, all_fps = args
    if i == 0:
        return []

    # Similarity vs. all previous molecules (lower triangle)
    sims = DataStructs.BulkTanimotoSimilarity(fp_i, all_fps[:i])
    # Convert similarity to distance
    return [1.0 - x for x in sims]


# --- MAIN CLASS ---

class DiversitySelector:
    """
    Implements the workflow for:
      - loading molecules,
      - generating fingerprints,
      - clustering with Butina,
      - selecting representatives,
      - exporting results,
      - and visualization.
    """

    def __init__(self, input_file: str, output_file: str, n_cores: int = 4):
        self.input_file = input_file
        self.output_file = output_file
        self.n_cores = n_cores if n_cores > 0 else multiprocessing.cpu_count()
        self.mols: List[Chem.Mol] = []
        self.fps: List[DataStructs.ExplicitBitVect] = []
        self.clusters: Tuple = ()
        self.selected_indices: List[int] = []

    def load_data(self) -> None:
        """Loads an SDF file and filters invalid molecules."""
        logger.info(f"Loading molecules from: {self.input_file}")
        if not self._check_file_exists():
            sys.exit(1)

        suppl = Chem.SDMolSupplier(self.input_file)
        self.mols = [x for x in suppl if x is not None]
        logger.info(f"Successfully loaded {len(self.mols)} valid molecules.")

    def generate_fingerprints(self) -> None:
        """Generates fingerprints in parallel."""
        logger.info(f"Generating fingerprints using {self.n_cores} cores...")
        start = time.time()

        with multiprocessing.Pool(self.n_cores) as pool:
            raw_fps = pool.map(_generate_fp_worker, self.mols)

        # Filter out entries that failed fingerprint generation
        valid_data = [(m, fp) for m, fp in zip(self.mols, raw_fps) if fp is not None]
        self.mols, self.fps = zip(*valid_data)
        self.mols = list(self.mols)
        self.fps = list(self.fps)

        logger.info(f"Fingerprint generation complete. Time: {time.time() - start:.2f}s")

    def cluster_data(self, cutoff: float) -> None:
        """
        Performs Butina clustering based on Tanimoto distance.

        Args:
            cutoff (float): Distance threshold (distance = 1 - similarity).
        """
        n_fps = len(self.fps)
        logger.info(f"Computing the distance matrix for {n_fps} molecules...")

        # Prepare tasks for parallel processing
        tasks = [(i, self.fps[i], self.fps) for i in range(1, n_fps)]

        dists = []
        with multiprocessing.Pool(self.n_cores) as pool:
            results = pool.map(_calc_dist_worker, tasks)

        for row in results:
            dists.extend(row)

        logger.info(f"Running Butina clustering (cutoff: {cutoff})...")
        self.clusters = Butina.ClusterData(dists, n_fps, cutoff, isDistData=True)
        logger.info(f"Identified {len(self.clusters)} distinct clusters.")

    def select_representatives(self, n_selection: int) -> None:
        """Selects cluster centroids until the requested target count is reached."""
        logger.info(f"Selecting {n_selection} representative compounds...")
        self.selected_indices = []

        # Select centroids from each cluster
        for cluster in self.clusters:
            self.selected_indices.append(cluster[0])  # cluster[0] is the centroid
            if len(self.selected_indices) >= n_selection:
                break

        logger.info(f"Final selection count: {len(self.selected_indices)}")

    def save_results(self) -> None:
        """Writes selected molecules to an SDF and exports a summary TXT report."""
        # 1) Save SDF
        logger.info(f"Saving selection to {self.output_file}...")
        writer = Chem.SDWriter(self.output_file)

        for idx in self.selected_indices:
            mol = self.mols[idx]

            # Linear search to assign Cluster_ID (centroid-based assignment)
            c_id = -1
            for i, clust in enumerate(self.clusters):
                if clust[0] == idx:
                    c_id = i
                    break

            mol.SetProp("Cluster_ID", str(c_id))
            mol.SetProp("Selection_Method", "Butina_Centroid")
            writer.write(mol)

        writer.close()

        # 2) Save statistics report (TXT)
        report_file = self.output_file.rsplit(".", 1)[0] + "_report.txt"
        logger.info(f"Saving statistics report to {report_file}...")

        total_mols = len(self.mols)
        selected_mols = len(self.selected_indices)
        total_clusters = len(self.clusters)
        singletons = sum(1 for c in self.clusters if len(c) == 1)

        selection_percentage = (selected_mols / total_mols) * 100 if total_mols > 0 else 0

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=========================================\n")
            f.write("          DIVERSITY SELECTION REPORT\n")
            f.write("=========================================\n\n")
            f.write(f"Total Molecules Processed: {total_mols}\n")
            f.write(f"Total Clusters Found:      {total_clusters}\n")
            f.write(f"Singletons (Unique Mols):  {singletons}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Selected Representatives:  {selected_mols}\n")
            f.write(f"Selection Percentage:      {selection_percentage:.2f}%\n")
            f.write("=========================================\n")

        logger.info("Save complete.")

    def visualize(self, output_prefix: str) -> None:
        """Generates analysis plots (PCA projection and cluster-size distribution)."""
        logger.info("Generating visualization plots...")
        sns.set_context("paper")

        self._plot_pca(output_prefix)
        self._plot_cluster_sizes(output_prefix)

    # --- INTERNAL PLOTTING HELPERS ---

    def _plot_pca(self, prefix: str) -> None:
        logger.info("Plotting PCA...")

        # Light grid styling for improved readability
        sns.set_style(
            "whitegrid",
            {
                "grid.color": ".92",
                "grid.linestyle": "-",
                "axes.edgecolor": ".8",
            },
        )

        X = []
        for fp in self.fps:
            arr = np.zeros((1,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            X.append(arr)
        X = np.array(X)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)

        plt.figure(figsize=(10, 8))

        # Full library (background)
        plt.scatter(
            coords[:, 0],
            coords[:, 1],
            c="#E0E0E0",
            alpha=0.6,
            s=15,
            label="Full Library",
        )

        # Selected representatives
        sel_coords = coords[self.selected_indices]
        plt.scatter(
            sel_coords[:, 0],
            sel_coords[:, 1],
            c="#E63946",
            marker="o",
            s=40,
            edgecolors="k",
            linewidth=0.5,
            label="Selected Representatives",
        )

        var_expl = pca.explained_variance_ratio_
        plt.xlabel(f"PC1 ({var_expl[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({var_expl[1]:.1%} variance)")
        plt.title(
            "Chemical Space Visualization (PCA)\n"
            f"N_total={len(self.fps)} | N_selected={len(self.selected_indices)}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{prefix}_pca_space.png", dpi=300)
        plt.close()

    def _plot_cluster_sizes(self, prefix: str) -> None:
        logger.info("Plotting cluster sizes...")

        # No grid for the bar plot
        sns.set_style("white")

        cluster_sizes = [len(c) for c in self.clusters]
        top_n = min(30, len(cluster_sizes))

        df_plot = pd.DataFrame(
            {
                "Rank": list(range(1, top_n + 1)),
                "Size": cluster_sizes[:top_n],
            }
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df_plot,
            x="Rank",
            y="Size",
            hue="Rank",
            palette="viridis",
            legend=False,
        )

        sns.despine()
        plt.xlabel("Cluster Rank")
        plt.ylabel("Cluster Size (Compounds)")
        plt.title(f"Top {top_n} Largest Clusters")
        plt.tight_layout()
        plt.savefig(f"{prefix}_cluster_dist.png", dpi=300)
        plt.close()

    def _check_file_exists(self) -> bool:
        import os

        if not os.path.exists(self.input_file):
            logger.error(f"Input file not found: {self.input_file}")
            return False
        return True


# --- RUN CONFIGURATION ---

if __name__ == "__main__":

    # ==========================================
    # USER CONFIGURATION (EDIT THIS SECTION)
    # ==========================================

    INPUT_SDF = "FILE_NAME.sdf"  		       # Input SDF filename
    OUTPUT_SDF = "FILE_NAME_final_selection.sdf"       # Output SDF filename

    TARGET_COUNT = 500        # Target number of compounds to select
    CUTOFF_DISTANCE = 0.65    # Butina distance cutoff (e.g., 0.35 distance = 0.65 similarity)

    CPU_CORES = 4             # Number of CPU cores to use (-1 for all)

    # ==========================================
    # EXECUTION (DO NOT MODIFY BELOW)
    # ==========================================

    print("\n--- Configuration ---")
    print(f"Input:  {INPUT_SDF}")
    print(f"Target: {TARGET_COUNT} molecules")
    print(f"Cores:  {CPU_CORES}")
    print("-" * 30)

    # Workflow execution
    selector = DiversitySelector(
        input_file=INPUT_SDF,
        output_file=OUTPUT_SDF,
        n_cores=CPU_CORES,
    )

    selector.load_data()
    selector.generate_fingerprints()
    selector.cluster_data(cutoff=CUTOFF_DISTANCE)
    selector.select_representatives(n_selection=TARGET_COUNT)
    selector.save_results()

    # Visualization
    plot_prefix = OUTPUT_SDF.rsplit(".", 1)[0]
    selector.visualize(plot_prefix)

    logger.info("All steps completed successfully.")

