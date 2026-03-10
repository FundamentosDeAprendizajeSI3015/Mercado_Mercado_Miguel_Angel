import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# --------------------------------------------------------------------
# 1. Lectura segura del dataset grande
# --------------------------------------------------------------------

# Usamos el archivo grande en formato Excel
DATA_PATH = "dataset_sintetico_FIRE_UdeA.xlsx"

# Leer Excel
df = pd.read_excel(DATA_PATH)

# Normalizar nombres de columnas (quitar espacios, pasar a minúsculas)
df.columns = df.columns.str.strip().str.lower()

# Si el Excel tiene una sola columna con todo el CSV pegado,
# la separamos en múltiples columnas usando la primera fila como datos.
if df.shape[1] == 1 and "," in df.columns[0]:
    header_str = df.columns[0]
    col_name = df.columns[0]
    # Separar cada fila por coma
    df_split = df[col_name].astype(str).str.split(",", expand=True)
    df_split.columns = [h.strip().lower() for h in header_str.split(",")]
    df = df_split

# Convertir columnas numéricas desde texto (todo menos 'label' inicialmente)
for col in df.columns:
    if col != "label":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Asegurar tipo de dato de año si existe la columna 'anio'
has_anio = "anio" in df.columns
if has_anio:
    df["anio"] = df["anio"].astype(int)

target_col = "label"
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].copy()
y = df[target_col].astype(int)


# --------------------------------------------------------------------
# 2. EDA ligero y robusto para muchos datos
# --------------------------------------------------------------------

print("----- Info general del dataframe -----")
print(df.info())

print("\n----- Descriptivos de variables numéricas (primeras 50k filas si aplica) -----")
num_df = df.select_dtypes(include=[np.number])
if len(num_df) > 50_000:
    print(num_df.sample(50_000, random_state=42).describe())
else:
    print(num_df.describe())

print("\n----- Valores faltantes por columna -----")
print(df.isna().sum())

sns.set(style="whitegrid")

# Para evitar plots enormes, muestreamos un subconjunto razonable
sample_for_plots = df
if len(df) > 20_000:
    sample_for_plots = df.sample(20_000, random_state=42)

# Distribución de la variable objetivo
plt.figure(figsize=(5, 4))
sns.countplot(x=target_col, data=sample_for_plots)
plt.title("Distribución de la variable objetivo (label)")
plt.savefig("eda_large_label_balance.png", dpi=300)
plt.close()

for col in ["liquidez", "dias_efectivo", "cfo"]:
    if col in df.columns:
        plt.figure(figsize=(5, 4))
        sns.histplot(sample_for_plots[col].dropna(), kde=False, bins=40)
        plt.title(f"Distribución de {col} (muestra)")
        plt.savefig(f"eda_large_hist_{col}.png", dpi=300)
        plt.close()

# Matriz de correlación sobre muestra y variables numéricas principales
num_cols_eda = num_df.columns.tolist()
if len(sample_for_plots) > 10_000:
    corr_sample = sample_for_plots[num_cols_eda].sample(10_000, random_state=42)
else:
    corr_sample = sample_for_plots[num_cols_eda]

corr = corr_sample.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Matriz de correlación (muestra)")
plt.tight_layout()
plt.savefig("eda_large_correlacion_numericas.png", dpi=300)
plt.close()


# --------------------------------------------------------------------
# 3. Partición train / test (temporal si hay 'anio', aleatoria si no)
# --------------------------------------------------------------------

from sklearn.model_selection import train_test_split

if has_anio:
    train_mask = df["anio"] <= 2022
    test_mask = df["anio"] > 2022

    X_train = X[train_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)

    X_test = X[test_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

print("\nTamaño train:", X_train.shape, " | Tamaño test:", X_test.shape)


# --------------------------------------------------------------------
# 4. Preprocesamiento escalable
# --------------------------------------------------------------------

cat_cols = ["unidad"] if "unidad" in X.columns else []
num_cols = [c for c in X.columns if c not in cat_cols]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

transformers = [("num", numeric_pipeline, num_cols)]
if cat_cols:
    transformers.append(("cat", categorical_pipeline, cat_cols))

preprocess = ColumnTransformer(transformers=transformers)


# --------------------------------------------------------------------
# 5. Modelos (pensados para más datos)
# --------------------------------------------------------------------

gb_clf = GradientBoostingClassifier(random_state=42)
rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)

pipe_gb = Pipeline(steps=[("prep", preprocess), ("model", gb_clf)])
pipe_rf = Pipeline(steps=[("prep", preprocess), ("model", rf_clf)])

# Grid más pequeño para no explotar en datasets grandes
param_grid_gb = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3],
}

# Si el dataset es enorme, se puede muestrear para el grid search
X_train_for_cv = X_train
y_train_for_cv = y_train
if len(X_train) > 50_000:
    sample_idx = np.random.RandomState(42).choice(
        len(X_train), size=50_000, replace=False
    )
    X_train_for_cv = X_train.iloc[sample_idx]
    y_train_for_cv = y_train.iloc[sample_idx]
    print("\nUsando muestra de 50k filas para GridSearchCV.")

gb_search = GridSearchCV(
    pipe_gb,
    param_grid=param_grid_gb,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1,
)

gb_search.fit(X_train_for_cv, y_train_for_cv)
print("Mejores params GB:", gb_search.best_params_)

best_gb = gb_search.best_estimator_


# --------------------------------------------------------------------
# 6. Evaluación en test
# --------------------------------------------------------------------

def evaluar_modelo(nombre, modelo, X_tr, y_tr, X_te, y_te):
    modelo.fit(X_tr, y_tr)
    y_pred = modelo.predict(X_te)
    if hasattr(modelo, "predict_proba"):
        y_proba = modelo.predict_proba(X_te)[:, 1]
    else:
        # Fallback por si acaso
        y_proba = None

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    roc = roc_auc_score(y_te, y_proba) if y_proba is not None else np.nan

    print(f"\n{nombre} - Test")
    print("Accuracy:", acc)
    print("F1      :", f1)
    print("ROC-AUC :", roc)

    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Matriz de confusión - {nombre}")
    plt.tight_layout()
    plt.savefig(f"cm_{nombre.replace(' ', '_').lower()}.png", dpi=300)
    plt.close()

    return y_pred, y_proba


# Gradient Boosting optimizado
_, _ = evaluar_modelo(
    "Gradient Boosting (large)",
    best_gb,
    X_train,
    y_train,
    X_test,
    y_test,
)

# Random Forest de referencia
_, _ = evaluar_modelo(
    "Random Forest (large)",
    pipe_rf,
    X_train,
    y_train,
    X_test,
    y_test,
)


# --------------------------------------------------------------------
# 7. Importancia de variables y árbol interpretable
# --------------------------------------------------------------------

feature_names_rf = pipe_rf.named_steps["prep"].get_feature_names_out()
importances_rf = pipe_rf.named_steps["model"].feature_importances_

importances_series = pd.Series(importances_rf, index=feature_names_rf).sort_values(
    ascending=False
)

plt.figure(figsize=(10, 6))
importances_series.head(30).plot(kind="bar")
plt.ylabel("Importancia (Gini)")
plt.title("Top 30 variables más importantes - Random Forest (large)")
plt.tight_layout()
plt.savefig("rf_large_importancia_variables.png", dpi=300)
plt.close()


# Árbol de decisión sencillo sobre muestra para interpretar reglas
dt_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_imputer = SimpleImputer(strategy="median")

X_train_num = X_train[num_cols]
if len(X_train_num) > 50_000:
    sample_idx_dt = np.random.RandomState(42).choice(
        len(X_train_num), size=50_000, replace=False
    )
    X_train_num = X_train_num.iloc[sample_idx_dt]
    y_train_dt = y_train.iloc[sample_idx_dt]
else:
    y_train_dt = y_train

X_train_num_imputed = dt_imputer.fit_transform(X_train_num)
dt_clf.fit(X_train_num_imputed, y_train_dt)

plt.figure(figsize=(22, 10))
plot_tree(
    dt_clf,
    feature_names=num_cols,
    class_names=["No tensión", "Tensión"],
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.tight_layout()
plt.savefig("arbol_decision_FIRE_UdeA_large.png", dpi=300)
plt.close()


# --------------------------------------------------------------------
# 8. PCA 2D sobre una muestra para visualización
# --------------------------------------------------------------------

# Usamos el preprocesamiento de Random Forest y una muestra para evitar matrices enormes
if len(X) > 30_000:
    sample_idx_emb = np.random.RandomState(42).choice(
        len(X), size=30_000, replace=False
    )
    X_emb_source = X.iloc[sample_idx_emb]
    y_emb = y.iloc[sample_idx_emb]
else:
    X_emb_source = X
    y_emb = y

X_emb = pipe_rf.named_steps["prep"].transform(X_emb_source)

pca_2d = PCA(n_components=2, random_state=42)
X_pca = pca_2d.fit_transform(X_emb.toarray() if hasattr(X_emb, "toarray") else X_emb)

pca_df = pd.DataFrame(
    {
        "pc1": X_pca[:, 0],
        "pc2": X_pca[:, 1],
        "label": y_emb.values,
    }
)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=pca_df,
    x="pc1",
    y="pc2",
    hue="label",
    palette="coolwarm",
    s=10,
)
plt.title("PCA 2D del espacio de características (muestra, dataset grande)")
plt.tight_layout()
plt.savefig("pca_FIRE_UdeA_large.png", dpi=300)
plt.close()

