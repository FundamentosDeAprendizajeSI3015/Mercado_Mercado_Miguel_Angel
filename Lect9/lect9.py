# %% [markdown]
# # Agrupamiento (Clustering)
# **SI3015 - Fundamentos de Aprendizaje Automático**

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.impute import SimpleImputer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    PolynomialFeatures,
    FunctionTransformer,
)

# %%
# Definamos el "random_state" para que los resultados sean reproducibles:
random_state = 42

# %%
# Cambiemos la fuente de las gráficas de matplotlib:
plt.rc("font", family="serif", size=12)

base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (base_dir / "dataset_sintetico_FIRE_UdeA Lect9.csv").exists():
    base_dir = base_dir / "Lect9"

dataset_1_path = base_dir / "dataset_sintetico_FIRE_UdeA Lect9.csv"
dataset_2_path = base_dir / "dataset_sintetico_FIRE_UdeA_realista Lect9.csv"
output_dir = base_dir


def cargar_dataset(path):
    df = pd.read_csv(path)
    features = df.drop(columns=["label"], errors="ignore").copy()
    return df, features


def construir_preprocessor(features):
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []

    if numeric_columns:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_columns))

    if categorical_columns:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_columns))

    return ColumnTransformer(transformers=transformers)


def obtener_columnas_plot(features):
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_columns) < 2:
        raise ValueError("Se necesitan al menos dos columnas numéricas para graficar.")
    return numeric_columns[:2]


def graficar_dataset(features, labels=None, title=None):
    x_col, y_col = obtener_columnas_plot(features)

    fig, ax = plt.subplots()
    ax.scatter(features[x_col], features[y_col], c=labels)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if title is not None:
        ax.set_title(title)
    fig.set_size_inches(5 * 1.6, 5)
    return fig, ax


def guardar_figura(fig, nombre_archivo):
    ruta_salida = output_dir / nombre_archivo
    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    print(f"Gráfica guardada en: {ruta_salida}")

# %% [markdown]
# ## Ejemplo Práctico
# Para los datasets planteados realice el siguiente proceso de clustering:
# - Halle y grafique los clusters resultantes de aplicar K-medias con $K = 2$.
# - Encuentre un buen valor de K mediante el método del codo y grafique los nuevos clusters.
# - Halle la inercia en cada caso.

# %% [markdown]
# ### Dataset 1

# %%
dataset_1_df, data = cargar_dataset(dataset_1_path)
dataset_1_df.head()

fig, ax = graficar_dataset(data, title="Dataset 1: primeras dos variables numéricas")
guardar_figura(fig, "dataset_1_variables_numericas.png")

# %%
# Definir el pipeline de pre-procesamiento
preprocessor = construir_preprocessor(data)

# %%
data.columns.tolist()

# %%
# Definimos el Pipeline de clustering con K = 2
clu_kmeans = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("clustering", KMeans(n_clusters=2, random_state=random_state, n_init=10)),
    ]
)

# %%
# Entrenamos
clu_kmeans.fit(data)
print(f'con K = 2: la inercia es {clu_kmeans["clustering"].inertia_}')

# %%
fig, ax = graficar_dataset(
    data,
    labels=clu_kmeans["clustering"].labels_,
    title="Dataset 1: K-means con K = 2",
)
guardar_figura(fig, "dataset_1_kmeans_k2.png")

# %%
inert = []
k_range = list(range(1, 11))
for k in k_range:
    clu_kmeans = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clustering", KMeans(n_clusters=k, random_state=random_state, n_init=10)),
        ]
    )
    clu_kmeans.fit(data)
    inert.append(clu_kmeans["clustering"].inertia_)

fig, ax = plt.subplots()
ax.plot(k_range, inert)
ax.set_xlabel("Número de clusters (K)")
ax.set_ylabel("Inercia")
ax.set_title("Dataset 1: método del codo")
fig.set_size_inches(5 * 1.6, 5)
guardar_figura(fig, "dataset_1_metodo_codo.png")

# %% [markdown]
# Daado que el "codo" está en $K = 2$, no aplicaremos K-medias de nuevo.

# %% [markdown]
# ### Dataset 2

# %%
dataset_2_df, data = cargar_dataset(dataset_2_path)
dataset_2_df.head()

fig, ax = graficar_dataset(data, title="Dataset 2: primeras dos variables numéricas")
guardar_figura(fig, "dataset_2_variables_numericas.png")

# %%
preprocessor = construir_preprocessor(data)

# %%
# Definimos el Pipeline de clustering con K = 2
clu_kmeans = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("clustering", KMeans(n_clusters=2, random_state=random_state, n_init=10)),
    ]
)
clu_kmeans.fit(data)
print(f'con K = 2: la inercia es {clu_kmeans["clustering"].inertia_}')

# %%
fig, ax = graficar_dataset(
    data,
    labels=clu_kmeans["clustering"].labels_,
    title="Dataset 2: K-means con K = 2",
)
guardar_figura(fig, "dataset_2_kmeans_k2.png")

# %%
inert = []
k_range = list(range(1, 11))
for k in k_range:
    clu_kmeans = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clustering", KMeans(n_clusters=k, random_state=random_state, n_init=10)),
        ]
    )
    clu_kmeans.fit(data)
    inert.append(clu_kmeans["clustering"].inertia_)

fig, ax = plt.subplots()
ax.plot(k_range, inert)
ax.set_xlabel("Número de clusters (K)")
ax.set_ylabel("Inercia")
ax.set_title("Dataset 2: método del codo")
fig.set_size_inches(5 * 1.6, 5)
guardar_figura(fig, "dataset_2_metodo_codo.png")

# %% [markdown]
# El "codo" está en $K = 4$.

# %%
# Definimos el Pipeline de clustering con K = 4
clu_kmeans = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("clustering", KMeans(n_clusters=4, random_state=random_state, n_init=10)),
    ]
)
clu_kmeans.fit(data)
print(f'con K = 4: la inercia es {clu_kmeans["clustering"].inertia_}')

# %%
fig, ax = graficar_dataset(
    data,
    labels=clu_kmeans["clustering"].labels_,
    title="Dataset 2: K-means con K = 4",
)
guardar_figura(fig, "dataset_2_kmeans_k4.png")

# %%
# Definimos el Pipeline de clustering con DBSCAN
# Ajuste `eps` si desea más o menos sensibilidad en los clusters.
clu_dbscan = Pipeline(
    steps=[("preprocessor", preprocessor), ("clustering", DBSCAN(eps=0.5, min_samples=5))]
)
clu_dbscan.fit(data)

# %%
fig, ax = graficar_dataset(
    data,
    labels=clu_dbscan["clustering"].labels_,
    title="Dataset 2: DBSCAN",
)
guardar_figura(fig, "dataset_2_dbscan.png")

# %%
np.unique(clu_dbscan["clustering"].labels_, return_counts=True)
