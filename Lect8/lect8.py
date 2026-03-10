import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Leer dataset
df = pd.read_csv("dataset_sintetico_FIRE_UdeA_realista.csv")

# 2. Definir variable objetivo (ya viene como 'label')
target_col = "label"

# 3. Features: todas menos label
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].copy()
y = df[target_col].astype(int)

from sklearn.model_selection import train_test_split

# Asegurar tipo de dato de año
df["anio"] = df["anio"].astype(int)

train_mask = df["anio"] <= 2022
test_mask  = df["anio"] >  2022

X_train = X[train_mask].reset_index(drop=True)
y_train = y[train_mask].reset_index(drop=True)

X_test  = X[test_mask].reset_index(drop=True)
y_test  = y[test_mask].reset_index(drop=True)

X_train.shape, X_test.shape

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA

# Separar columnas numéricas y categóricas
cat_cols = ["unidad"]  # categórica
num_cols = [c for c in X.columns if c not in cat_cols]

# 0) EDA básico
print("----- Info general del dataframe -----")
print(df.info())

print("\n----- Descriptivos de variables numéricas -----")
print(df.describe())

print("\n----- Valores faltantes por columna -----")
print(df.isna().sum())

sns.set(style="whitegrid")

# Distribución de la variable objetivo
plt.figure(figsize=(5, 4))
sns.countplot(x=target_col, data=df)
plt.title("Distribución de la variable objetivo (label)")
plt.savefig("eda_label_balance.png", dpi=300)
plt.close()

# Histogramas de algunas variables clave, si existen
for col in ["liquidez", "dias_efectivo", "cfo"]:
    if col in df.columns:
        plt.figure(figsize=(5, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribución de {col}")
        plt.savefig(f"eda_hist_{col}.png", dpi=300)
        plt.close()

# distribuciones por etiqueta: violin, box y strip/swarm
for col in ["liquidez", "dias_efectivo", "cfo"]:
    if col in df.columns:
        # violin
        plt.figure(figsize=(6, 4))
        sns.violinplot(x=target_col, y=col, data=df, palette="Set2")
        plt.title(f"Violin plot de {col} por label")
        plt.savefig(f"eda_violin_{col}.png", dpi=300)
        plt.close()
        # box
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=target_col, y=col, data=df, palette="Set1")
        plt.title(f"Box plot de {col} por label")
        plt.savefig(f"eda_box_{col}.png", dpi=300)
        plt.close()
        # strip/swarm
        plt.figure(figsize=(6, 4))
        sns.stripplot(x=target_col, y=col, data=df, color="black", alpha=0.5)
        plt.title(f"Strip plot de {col} por label")
        plt.savefig(f"eda_strip_{col}.png", dpi=300)
        plt.close()
        plt.figure(figsize=(6, 4))
        sns.swarmplot(x=target_col, y=col, data=df, palette="pastel")
        plt.title(f"Swarm plot de {col} por label")
        plt.savefig(f"eda_swarm_{col}.png", dpi=300)
        plt.close()

# Matriz de correlación de variables numéricas
num_cols_eda = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[num_cols_eda].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Matriz de correlación de variables numéricas")
plt.tight_layout()
plt.savefig("eda_correlacion_numericas.png", dpi=300)
plt.close()

# Preprocesamiento con imputación de faltantes
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ]
)

gb_clf = GradientBoostingClassifier(random_state=42)
rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

pipe_gb = Pipeline(steps=[("prep", preprocess), ("model", gb_clf)])
pipe_rf = Pipeline(steps=[("prep", preprocess), ("model", rf_clf)])

from sklearn.model_selection import GridSearchCV

param_grid_gb = {
    "model__n_estimators": [100, 200, 400],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__subsample": [0.8, 1.0],
}

gb_search = GridSearchCV(
    pipe_gb,
    param_grid=param_grid_gb,
    scoring="f1",
    cv=3,
    n_jobs=-1,
)

gb_search.fit(X_train, y_train)

print("Mejores params GB:", gb_search.best_params_)
best_gb = gb_search.best_estimator_

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# 1) Gradient Boosting optimizado
y_pred_gb = best_gb.predict(X_test)
y_proba_gb = best_gb.predict_proba(X_test)[:, 1]

print("Gradient Boosting - Test")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("F1      :", f1_score(y_test, y_pred_gb))
print("ROC-AUC :", roc_auc_score(y_test, y_proba_gb))

cm_gb = confusion_matrix(y_test, y_pred_gb)
disp_gb = ConfusionMatrixDisplay(cm_gb)
disp_gb.plot()
plt.title("Matriz de confusión - Gradient Boosting")
plt.savefig("cm_gradient_boosting.png", dpi=300)
plt.close()

# 2) Random Forest como modelo base fuerte
pipe_rf.fit(X_train, y_train)
y_pred_rf = pipe_rf.predict(X_test)
y_proba_rf = pipe_rf.predict_proba(X_test)[:, 1]

print("\nRandom Forest - Test")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1      :", f1_score(y_test, y_pred_rf))
print("ROC-AUC :", roc_auc_score(y_test, y_proba_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(cm_rf)
disp_rf.plot()
plt.title("Matriz de confusión - Random Forest")
plt.savefig("cm_random_forest.png", dpi=300)
plt.close()

# 3) Importancia de variables del Random Forest

# Obtener nombres de features después del preprocesamiento (incluye dummies)
feature_names_rf = pipe_rf.named_steps["prep"].get_feature_names_out()
importances_rf = pipe_rf.named_steps["model"].feature_importances_

importances_series = pd.Series(importances_rf, index=feature_names_rf).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importances_series.head(20).plot(kind="bar")
plt.ylabel("Importancia (Gini)")
plt.title("Top 20 variables más importantes - Random Forest")
plt.tight_layout()
plt.savefig("rf_importancia_variables.png", dpi=300)
plt.close()

# 4) Árbol de decisión interpretable y exportar imagen

# Entrenar un árbol de decisión poco profundo para poder visualizarlo claramente
dt_clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# Imputar faltantes en variables numéricas antes de entrenar el árbol
dt_imputer = SimpleImputer(strategy="median")
X_train_num_imputed = dt_imputer.fit_transform(X_train[num_cols])

dt_clf.fit(X_train_num_imputed, y_train)

plt.figure(figsize=(20, 10))
plot_tree(
    dt_clf,
    feature_names=num_cols,
    class_names=["No tensión", "Tensión"],
    filled=True,
    rounded=True,
    fontsize=8,
)

plt.tight_layout()
plt.savefig("arbol_decision_FIRE_UdeA.png", dpi=300)
plt.close()

# 5) PCA 2D para visualizar el espacio de representación (alternativa a UMAP)

# Transformar todo el dataset con el preprocesamiento aprendido
X_emb = pipe_rf.named_steps["prep"].transform(X)

pca_2d = PCA(n_components=2, random_state=42)
X_pca = pca_2d.fit_transform(X_emb)

pca_df = pd.DataFrame(
    {
        "pc1": X_pca[:, 0],
        "pc2": X_pca[:, 1],
        "label": y.values,
    }
)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=pca_df,
    x="pc1",
    y="pc2",
    hue="label",
    palette="coolwarm",
)
plt.title("PCA 2D del espacio de características (coloreado por label)")
plt.tight_layout()
plt.savefig("pca_fire_udea.png", dpi=300)
plt.close()

# ------- curvas comparativas y calibración -------
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve

# ROC comparativa
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_proba_gb)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
roc_auc_gb = auc(fpr_gb, tpr_gb)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(figsize=(6, 5))
plt.plot(fpr_gb, tpr_gb, label=f"GB (AUC={roc_auc_gb:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={roc_auc_rf:.2f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC comparativa")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_comparativa.png", dpi=300)
plt.close()

# Precision-Recall comparativa
prec_gb, rec_gb, _ = precision_recall_curve(y_test, y_proba_gb)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_proba_rf)
pr_auc_gb = auc(rec_gb, prec_gb)
pr_auc_rf = auc(rec_rf, prec_rf)
plt.figure(figsize=(6, 5))
plt.plot(rec_gb, prec_gb, label=f"GB (AUC={pr_auc_gb:.2f})")
plt.plot(rec_rf, prec_rf, label=f"RF (AUC={pr_auc_rf:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall comparativa")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("pr_comparativa.png", dpi=300)
plt.close()

# Calibration plot
prob_true_gb, prob_pred_gb = calibration_curve(y_test, y_proba_gb, n_bins=10)
prob_true_rf, prob_pred_rf = calibration_curve(y_test, y_proba_rf, n_bins=10)
plt.figure(figsize=(6, 5))
plt.plot(prob_pred_gb, prob_true_gb, "s-", label="GB")
plt.plot(prob_pred_rf, prob_true_rf, "s-", label="RF")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration plot")
plt.legend()
plt.tight_layout()
plt.savefig("calibration_plot.png", dpi=300)
plt.close()
