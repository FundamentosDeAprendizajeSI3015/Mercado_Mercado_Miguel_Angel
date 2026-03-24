import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN


def main():
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, "dataset_sintetico_FIRE_UdeA Lect9 copy.csv")
    outdir = os.path.join(base, "visualizaciones_u_dea")
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(csv_path)
    y_label = df["label"].astype(int)

    Xdf = df.drop(columns=[c for c in df.columns if c.lower() == "label"])
    Xdf = Xdf.select_dtypes(include=[np.number])

    X = SimpleImputer(strategy="mean").fit_transform(Xdf)
    Xz = StandardScaler().fit_transform(X)

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    latent = gmm.fit_predict(Xz)

    aux = Xdf.copy()
    aux["latent"] = latent
    center = aux.groupby("latent")[["liquidez", "cfo", "dias_efectivo"]].mean()
    health_score = (
        center["liquidez"].rank(pct=True)
        + center["cfo"].rank(pct=True)
        + center["dias_efectivo"].rank(pct=True)
    )
    healthy_cluster = health_score.idxmax()
    true_class = np.where(latent == healthy_cluster, 1, 2)

    mask_l1 = y_label == 1
    total_l1 = int(mask_l1.sum())
    class1_in_l1 = int((true_class[mask_l1] == 1).sum())
    class2_in_l1 = int((true_class[mask_l1] == 2).sum())
    p1 = (100 * class1_in_l1 / total_l1) if total_l1 else 0.0
    p2 = (100 * class2_in_l1 / total_l1) if total_l1 else 0.0

    suspected = ((y_label == 1) & (true_class == 2)) | ((y_label == 0) & (true_class == 1))

    db = DBSCAN(eps=3.1410, min_samples=5).fit(Xz)
    db_labels = db.labels_
    core_mask = np.zeros(len(Xz), dtype=bool)
    core_mask[db.core_sample_indices_] = True

    sus_n = int(suspected.sum())
    if sus_n > 0:
        sus_core = int((suspected & core_mask).sum())
        sus_border = int((suspected & (db_labels != -1) & (~core_mask)).sum())
        sus_out = int((suspected & (db_labels == -1)).sum())
    else:
        sus_core = sus_border = sus_out = 0

    sns.set_theme(style="whitegrid", context="talk")

    pca2 = PCA(n_components=2, random_state=42)
    X2 = pca2.fit_transform(Xz)
    plot_df = pd.DataFrame(
        {
            "PC1": X2[:, 0],
            "PC2": X2[:, 1],
            "ClaseLatente": true_class,
            "Etiqueta": y_label,
            "Sospechoso": suspected,
        }
    )
    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="ClaseLatente",
        style="Etiqueta",
        palette="Set2",
        s=85,
        alpha=0.85,
    )
    plt.title("UdeA: PCA 2D por clase latente (1/2) y etiqueta original")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "01_pca2d_clase_latente.png"), dpi=300)
    plt.close()

    pca3 = PCA(n_components=3, random_state=42)
    X3 = pca3.fit_transform(Xz)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    colors = np.where(true_class == 1, "#1f77b4", "#d62728")
    ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=colors, s=38, alpha=0.8)
    ax.set_title("UdeA: PCA 3D (azul=Clase 1, rojo=Clase 2)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "02_pca3d_clases.png"), dpi=300)
    plt.close()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto")
    Xt = tsne.fit_transform(Xz)
    tsdf = pd.DataFrame(
        {
            "tSNE1": Xt[:, 0],
            "tSNE2": Xt[:, 1],
            "ClaseLatente": true_class,
            "Sospechoso": suspected,
        }
    )
    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        data=tsdf,
        x="tSNE1",
        y="tSNE2",
        hue="ClaseLatente",
        style="Sospechoso",
        palette="coolwarm",
        s=80,
        alpha=0.9,
    )
    plt.title("UdeA: t-SNE 2D con puntos sospechosos resaltados")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "03_tsne_sospechosos.png"), dpi=300)
    plt.close()

    sample_n = min(220, len(Xdf))
    sm = Xdf.sample(sample_n, random_state=42).copy()
    sm["ClaseLatente"] = true_class[sm.index]
    cols = [
        c
        for c in ["liquidez", "dias_efectivo", "cfo", "hhi_fuentes", "gastos_personal"]
        if c in sm.columns
    ]
    if len(cols) >= 3:
        g = sns.pairplot(
            sm[cols + ["ClaseLatente"]],
            hue="ClaseLatente",
            corner=True,
            diag_kind="kde",
            plot_kws={"alpha": 0.6, "s": 28},
        )
        g.fig.suptitle("UdeA: Pairplot de variables clave por clase latente", y=1.02)
        g.savefig(os.path.join(outdir, "04_pairplot_variables_clave.png"), dpi=250)
        plt.close("all")

    metrics = {
        "total_l1": total_l1,
        "class1_in_l1": class1_in_l1,
        "class2_in_l1": class2_in_l1,
        "p_class1_in_l1": round(p1, 2),
        "p_class2_in_l1": round(p2, 2),
        "suspected_n": sus_n,
        "suspected_core": sus_core,
        "suspected_border": sus_border,
        "suspected_outliers": sus_out,
        "suspected_core_pct": round((100 * sus_core / sus_n) if sus_n else 0, 2),
        "suspected_border_pct": round((100 * sus_border / sus_n) if sus_n else 0, 2),
        "suspected_outliers_pct": round((100 * sus_out / sus_n) if sus_n else 0, 2),
        "pca2_explained_pct": round(float(pca2.explained_variance_ratio_.sum() * 100), 2),
        "visual_dir": outdir,
    }

    metrics_path = os.path.join(base, "metricas_respuestas_u_dea.json")
    pd.Series(metrics).to_json(metrics_path, force_ascii=False, indent=2)
    print("METRICS_PATH", metrics_path)
    print(metrics)


if __name__ == "__main__":
    main()
