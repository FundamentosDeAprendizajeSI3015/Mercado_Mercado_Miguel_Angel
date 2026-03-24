# ==========================================================
# Clustering Pipeline EXTENDED
# Hierarchical, DBSCAN, HDBSCAN, UMAP
# + k-distance plot
# + Clustering stability analysis
# ==========================================================

# -----------------------------
# 1. LIBRARIES
# -----------------------------
import numpy as np
import pandas as pd
import os
import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from scipy.cluster.hierarchy import linkage, dendrogram
import hdbscan

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[WARN] umap-learn not available (requires Python ≤ 3.13). UMAP plots will be skipped.")

sns.set(style="whitegrid", context="talk")

# -----------------------------
# 2. DATA LOADING
# -----------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded data: {df.shape[0]} samples, {df.shape[1]} columns")
    return df

# -----------------------------
# 3. PREPROCESSING
# -----------------------------

def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """
    Drops the 'label' column (target) and any non-numeric columns,
    then applies StandardScaler.
    """
    # Drop target label if present
    cols_to_drop = [c for c in df.columns if c.lower() == 'label']
    df = df.drop(columns=cols_to_drop)

    # Keep only numeric columns (drops e.g. 'unidad', 'anio' strings)
    df_numeric = df.select_dtypes(include=[np.number])
    dropped = set(df.columns) - set(df_numeric.columns)
    if dropped:
        print(f"[INFO] Non-numeric columns dropped: {dropped}")
    print(f"[INFO] Features used for clustering: {list(df_numeric.columns)}")

    # Impute NaN values with column mean before scaling
    n_nan = df_numeric.isna().sum().sum()
    if n_nan > 0:
        print(f"[INFO] Imputing {n_nan} NaN values (mean strategy)")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(df_numeric.values)

    scaler = StandardScaler()
    return scaler.fit_transform(X_imputed)

# -----------------------------
# 4. k-DISTANCE PLOT (DBSCAN)
# -----------------------------

def k_distance_plot(X: np.ndarray, k: int = 5, save_path: str = None):
    """
    Automatic k-distance plot for DBSCAN parameter selection.
    Typical choice: k = min_samples.
    Returns a suggested eps (elbow/knee of the curve).
    """
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, k-1])

    # Simple knee detection: largest second derivative
    diff2 = np.diff(np.diff(k_distances))
    knee_idx = np.argmax(diff2) + 2
    suggested_eps = k_distances[knee_idx]
    print(f"[INFO] Suggested eps (knee of k-distance): {suggested_eps:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.axhline(y=suggested_eps, color='red', linestyle='--',
                label=f'Suggested eps = {suggested_eps:.3f}')
    plt.xlabel('Sorted observations')
    plt.ylabel(f'{k}-distance')
    plt.title('k-distance plot (DBSCAN)')
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return suggested_eps

# -----------------------------
# 5. HIERARCHICAL CLUSTERING
# -----------------------------

def hierarchical_clustering(X: np.ndarray, method: str):
    return linkage(X, method=method)


def plot_dendrogram(Z, method: str, truncate_level: int = 40, save_path: str = None):
    plt.figure(figsize=(14, 6))
    dendrogram(Z, truncate_mode='level', p=truncate_level)
    plt.title(f'Dendrogram – {method.upper()}')
    plt.xlabel('Observations')
    plt.ylabel('Distance')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------
# 6. DBSCAN
# -----------------------------

def run_dbscan(X: np.ndarray, eps: float, min_samples: int):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)

# -----------------------------
# 7. HDBSCAN
# -----------------------------

def run_hdbscan(X: np.ndarray, min_cluster_size: int):
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(X)
    return labels

# -----------------------------
# 8. DIMENSIONALITY REDUCTION
# -----------------------------

def reduce_pca(X: np.ndarray):
    pca = PCA(n_components=2)
    X_red = pca.fit_transform(X)
    print(f"[INFO] PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    return X_red


def reduce_umap(X: np.ndarray):
    if not UMAP_AVAILABLE:
        print("[WARN] UMAP unavailable – returning None.")
        return None
    reducer = umap.UMAP(n_components=2, random_state=42)
    return reducer.fit_transform(X)

# -----------------------------
# 9. VISUALIZATION
# -----------------------------

def plot_clusters(X_2d, labels, title, save_path: str = None):
    df_plot = pd.DataFrame({
        'Dim1': X_2d[:, 0],
        'Dim2': X_2d[:, 1],
        'Cluster': labels.astype(str)
    })

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_plot,
        x='Dim1',
        y='Dim2',
        hue='Cluster',
        palette='tab10',
        s=70,
        alpha=0.85
    )
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def safe_filename(text: str) -> str:
    sanitized = re.sub(r'[^\w\-\s]', '', text, flags=re.UNICODE)
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    return sanitized

# -----------------------------
# 10. CLUSTERING STABILITY
# -----------------------------

def clustering_stability(X: np.ndarray, cluster_func, n_bootstrap: int = 20):
    """
    Estimates clustering stability using bootstrap
    and Adjusted Rand Index (ARI).
    """
    labels_ref = cluster_func(X)
    ari_scores = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_sample = X[idx]
        labels_sample = cluster_func(X_sample)

        # Align sizes for ARI
        ari = adjusted_rand_score(labels_ref[idx], labels_sample)
        ari_scores.append(ari)

    print(f"[INFO] Mean ARI stability: {np.mean(ari_scores):.3f}")
    print(f"[INFO] Std ARI stability: {np.std(ari_scores):.3f}")

# -----------------------------
# 11. PIPELINE
# -----------------------------

DATASETS = [
    {
        "path": "dataset_sintetico_FIRE_UdeA Lect9 copy.csv",
        "name": "FIRE_Sintetico",
        "dbscan_eps": 0.5,
        "dbscan_min_samples": 5,
        "hdbscan_min_cluster_size": 10,
    },
    {
        "path": "dataset_sintetico_FIRE_UdeA_realista Lect9 copy.csv",
        "name": "FIRE_Realista",
        "dbscan_eps": 0.5,
        "dbscan_min_samples": 5,
        "hdbscan_min_cluster_size": 10,
    },
]


def run_pipeline(cfg: dict):
    csv_path = os.path.join(os.path.dirname(__file__), cfg["path"])
    name = cfg["name"]
    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)

    eps = cfg["dbscan_eps"]
    min_s = cfg["dbscan_min_samples"]
    min_cs = cfg["hdbscan_min_cluster_size"]

    print(f"\n{'='*60}")
    print(f"  DATASET: {name}")
    print(f"{'='*60}")

    # Load & preprocess
    df = load_data(csv_path)
    X = preprocess_data(df)

    # k-distance plot – auto-estimate eps
    eps = k_distance_plot(
        X,
        k=min_s,
        save_path=os.path.join(output_dir, f"{safe_filename(name)}_k_distance.png")
    )
    print(f"[INFO] Using eps={eps:.4f} for DBSCAN")

    # Dimensionality reduction
    X_pca = reduce_pca(X)
    X_umap = reduce_umap(X)

    # Hierarchical
    for method in ['single', 'complete', 'average', 'ward']:
        Z = hierarchical_clustering(X, method)
        plot_dendrogram(
            Z,
            f"{name} – {method}",
            save_path=os.path.join(output_dir, f"{safe_filename(name)}_dendrogram_{method}.png")
        )

    # DBSCAN
    db_labels = run_dbscan(X, eps=eps, min_samples=min_s)
    n_clusters_db = len(set(db_labels) - {-1})
    print(f"[DBSCAN] clusters={n_clusters_db}, noise={np.sum(db_labels == -1)}")
    plot_clusters(
        X_pca,
        db_labels,
        f'DBSCAN + PCA  [{name}]',
        save_path=os.path.join(output_dir, f"{safe_filename(name)}_dbscan_pca.png")
    )
    if X_umap is not None:
        plot_clusters(
            X_umap,
            db_labels,
            f'DBSCAN + UMAP [{name}]',
            save_path=os.path.join(output_dir, f"{safe_filename(name)}_dbscan_umap.png")
        )

    # Stability DBSCAN
    print("[Stability] DBSCAN")
    clustering_stability(X, lambda X_: run_dbscan(X_, eps=eps, min_samples=min_s))

    # HDBSCAN
    hdb_labels = run_hdbscan(X, min_cluster_size=min_cs)
    n_clusters_hdb = len(set(hdb_labels) - {-1})
    print(f"[HDBSCAN] clusters={n_clusters_hdb}, noise={np.sum(hdb_labels == -1)}")
    plot_clusters(
        X_pca,
        hdb_labels,
        f'HDBSCAN + PCA  [{name}]',
        save_path=os.path.join(output_dir, f"{safe_filename(name)}_hdbscan_pca.png")
    )
    if X_umap is not None:
        plot_clusters(
            X_umap,
            hdb_labels,
            f'HDBSCAN + UMAP [{name}]',
            save_path=os.path.join(output_dir, f"{safe_filename(name)}_hdbscan_umap.png")
        )

    # Stability HDBSCAN
    print("[Stability] HDBSCAN")
    clustering_stability(X, lambda X_: run_hdbscan(X_, min_cluster_size=min_cs))


def main():
    for cfg in DATASETS:
        run_pipeline(cfg)


if __name__ == '__main__':
    main()
