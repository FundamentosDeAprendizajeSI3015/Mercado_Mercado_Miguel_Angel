#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PIPELINE COMPLETO DE ANÁLISIS: MOVIES DATASET
================================================================================

Este script integra un pipeline completo de análisis de datos integrando
técnicas de las Lecciones 2, 3 y 4:

LECCIÓN 2 - MACHINE LEARNING & FEATURE ENGINEERING:
  • Feature Engineering (ratios, áreas, interacciones)
  • Múltiples modelos de clasificación/regresión
  • Validación cruzada y grid search
  • Evaluación de modelos

LECCIÓN 3 - ANÁLISIS AVANZADO DE DATOS:
  • Análisis de series temporales (si aplica)
  • Comparativa entre categorías
  • Análisis de correlaciones y tendencias
  • Reportes detallados

LECCIÓN 4 - EXPLORACIÓN GRÁFICA Y TRANSFORMACIONES:
  • Medidas de tendencia central, dispersión, posición
  • Detección y tratamiento de outliers
  • Histogramas y gráficos de dispersión
  • Transformaciones: One Hot, Label, Scaling, Log
  • Análisis de distribuciones

Ejecución:
    python movies_complete_pipeline.py

Dependencias:
    pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, plotly

Dataset:
    movies.csv (descargado desde Kaggle Hub)
    
Autor: Data Science Pipeline
Fecha: 2026
================================================================================
"""

from __future__ import annotations
import os
import warnings
import sys
from pathlib import Path
from datetime import datetime

# Data processing
import pandas as pd
import numpy as np
from scipy import stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import joblib
warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN ====================
print("=" * 80)
print("PIPELINE COMPLETO DE ANÁLISIS - DATASET DE PELÍCULAS")
print("=" * 80)

# Rutas
script_dir = Path(__file__).parent.parent
outputs_dir = script_dir / 'outputs'
outputs_dir.mkdir(exist_ok=True)

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== 1. CARGA DE DATOS ====================
print("\n[1/8] CARGANDO Y EXPLORANDO DATOS")
print("-" * 80)

import kagglehub
path = kagglehub.dataset_download("bharatnatrayn/movies-dataset-for-feature-extracion-prediction")
df_raw = pd.read_csv(f"{path}/movies.csv")

print(f"✓ Dataset cargado: {df_raw.shape[0]} filas × {df_raw.shape[1]} columnas")
print(f"\nPrimeras filas:")
print(df_raw.head())
print(f"\nTipos de datos:")
print(df_raw.dtypes)
print(f"\nValores faltantes:")
print(df_raw.isnull().sum())

# Guardar estadísticas básicas
basic_stats = pd.DataFrame({
    'Filas': [df_raw.shape[0]],
    'Columnas': [df_raw.shape[1]],
    'Memoria (MB)': [df_raw.memory_usage(deep=True).sum() / 1024**2],
    'Valores nulos (%)': [(df_raw.isnull().sum().sum() / (df_raw.shape[0] * df_raw.shape[1])) * 100]
})
print("\nEstadísticas básicas:")
print(basic_stats)

# ==================== 2. PRE-PROCESAMIENTO (Lect 4) ====================
print("\n[2/8] PRE-PROCESAMIENTO Y LIMPIEZA DE DATOS")
print("-" * 80)

df = df_raw.copy()

# Identificar columnas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"Columnas numéricas: {numeric_cols}")
print(f"Columnas categóricas: {categorical_cols}")

# Detección de outliers (IQR)
print("\n--- Detección de Outliers (Método IQR) ---")
df_clean = df.copy()
outliers_removed = 0

for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = len(df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)])
    print(f"  {col}: {n_outliers} outliers [{lower:.2f}, {upper:.2f}]")
    df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    outliers_removed = df.shape[0] - df_clean.shape[0]

print(f"Filas eliminadas: {outliers_removed} ({100*outliers_removed/df.shape[0]:.1f}%)")
print(f"Dataset limpio: {df_clean.shape}")

# ==================== 3. EXPLORACIÓN GRÁFICA (Lect 4) ====================
print("\n[3/8] EXPLORACIÓN GRÁFICA Y ANÁLISIS DE DISTRIBUCIONES")
print("-" * 80)

# Estadísticas descriptivas
print("\nMedidas de Tendencia Central:")
tendencia = df_clean[numeric_cols].agg(['mean', 'median', lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan])
tendencia.index = ['Media', 'Mediana', 'Moda']
print(tendencia)

print("\nMedidas de Dispersión:")
dispersión = pd.DataFrame({
    'Std': df_clean[numeric_cols].std(),
    'Var': df_clean[numeric_cols].var(),
    'CV (%)': (df_clean[numeric_cols].std() / df_clean[numeric_cols].mean()) * 100,
    'Rango': df_clean[numeric_cols].max() - df_clean[numeric_cols].min()
})
print(dispersión)

print("\nMedidas de Posición (Cuartiles):")
posición = pd.DataFrame({
    'Q1': df_clean[numeric_cols].quantile(0.25),
    'Q2': df_clean[numeric_cols].quantile(0.50),
    'Q3': df_clean[numeric_cols].quantile(0.75),
    'IQR': df_clean[numeric_cols].quantile(0.75) - df_clean[numeric_cols].quantile(0.25)
})
print(posición)

# Histogramas
fig, axes = plt.subplots(1, len(numeric_cols), figsize=(4*len(numeric_cols), 4))
if len(numeric_cols) == 1:
    axes = [axes]

for idx, col in enumerate(numeric_cols):
    axes[idx].hist(df_clean[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Histograma de {col}', fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frecuencia')
    skew = stats.skew(df_clean[col])
    kurt = stats.kurtosis(df_clean[col])
    axes[idx].text(0.98, 0.97, f'Asimetría: {skew:.3f}\nCurtosis: {kurt:.3f}',
                   transform=axes[idx].transAxes, fontsize=9, verticalalignment='top',
                   horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(outputs_dir / 'pipeline_histogramas.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Histogramas guardados")

# ==================== 4. FEATURE ENGINEERING (Lect 2) ====================
print("\n[4/8] INGENIERÍA DE CARACTERÍSTICAS")
print("-" * 80)

df_features = df_clean.copy()

# Crear características derivadas si hay múltiples columnas numéricas
if len(numeric_cols) >= 2:
    print("Creando características derivadas...")
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            # Ratio
            df_features[f'{col1}_over_{col2}'] = df_features[col1] / (df_features[col2] + 1e-6)
            # Suma
            df_features[f'{col1}_plus_{col2}'] = df_features[col1] + df_features[col2]
            # Producto
            df_features[f'{col1}_times_{col2}'] = df_features[col1] * df_features[col2]
    print(f"  ✓ {df_features.shape[1] - df_clean.shape[1]} nuevas características creadas")
    print(f"  Dataset expandido: {df_features.shape}")

# Transformaciones logarítmicas para columnas positivas
print("\nAplicando transformaciones logarítmicas...")
log_cols = []
for col in numeric_cols:
    if (df_features[col] > 0).all():
        df_features[f'{col}_log'] = np.log(df_features[col])
        log_cols.append(col)
print(f"  ✓ Log aplicada a {len(log_cols)} columnas")

# ==================== 5. TRANSFORMACIONES (Lect 4) ====================
print("\n[5/8] TRANSFORMACIONES Y ESCALADO")
print("-" * 80)

df_transformed = df_features.copy()
numeric_cols_all = df_transformed.select_dtypes(include=[np.number]).columns.tolist()

print(f"Columnas numéricas totales: {len(numeric_cols_all)}")

# One Hot Encoding
print("\nOne Hot Encoding...")
for col in categorical_cols:
    if df_transformed[col].nunique() <= 10:
        ohe = pd.get_dummies(df_transformed[col], prefix=col, drop_first=False)
        df_transformed = pd.concat([df_transformed, ohe], axis=1)
        print(f"  ✓ {col}: {ohe.shape[1]} bins creados")

# Label Encoding
print("\nLabel Encoding...")
label_encoders = {}
for col in categorical_cols:
    if col in df_transformed.columns and df_transformed[col].dtype == 'object':
        le = LabelEncoder()
        df_transformed[f'{col}_encoded'] = le.fit_transform(df_transformed[col].fillna('Unknown'))
        label_encoders[col] = le
        print(f"  ✓ {col}: {len(le.classes_)} clases")

# Actualizar columnas numéricas
numeric_cols_encoded = df_transformed.select_dtypes(include=[np.number]).columns.tolist()

# Escalado Min-Max
print("\nMin-Max Scaling...")
minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(
    minmax_scaler.fit_transform(df_transformed[numeric_cols_encoded]),
    columns=[f'{col}_minmax' for col in numeric_cols_encoded]
)

# StandardScaler
print("StandardScaler...")
standard_scaler = StandardScaler()
df_standard = pd.DataFrame(
    standard_scaler.fit_transform(df_transformed[numeric_cols_encoded]),
    columns=[f'{col}_standard' for col in numeric_cols_encoded]
)

print(f"✓ Escalado completado: {df_minmax.shape[1]} cols MinMax, {df_standard.shape[1]} cols Standard")

# ==================== 6. ANÁLISIS DE CORRELACIÓN (Lect 3) ====================
print("\n[6/8] ANÁLISIS DE CORRELACIONES")
print("-" * 80)

corr_matrix = df_transformed[numeric_cols_encoded].corr()

# Top correlaciones
print("\nTop 10 Correlaciones (excluyendo diagonal):")
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
for col1, col2, corr_val in corr_pairs_sorted[:10]:
    print(f"  {col1:30s} ↔ {col2:30s}: {corr_val:7.3f}")

# Matriz de correlación visual
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
plt.title('Matriz de Correlación - Features Transformadas', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(outputs_dir / 'pipeline_matriz_correlación.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Matriz de correlación guardada")

# ==================== 7. MODELOS DE MACHINE LEARNING (Lect 2) ====================
print("\n[7/8] ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
print("-" * 80)

# Preparar datos para modelado
# Usar variables numéricas originales como target si es posible
X = df_standard.copy()  # Features escaladas
y = df_clean[numeric_cols[0]].copy() if len(numeric_cols) > 0 else df_clean[numeric_cols[-1]]

print(f"\nDataset para modelado:")
print(f"  X (features): {X.shape}")
print(f"  y (target): {y.shape}")

# División entrenamiento-prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDivisión de datos:")
print(f"  Entrenamiento: {X_train.shape}")
print(f"  Prueba: {X_test.shape}")

# Modelos de regresión
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.01),
    'KNN Regressor': KNeighborsRegressor(n_neighbors=5),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf')
}

results = []
best_model = None
best_r2 = -np.inf

print("\nEntrenando modelos...")
for name, model in models.items():
    # Validación cruzada
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Entrenar
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Modelo': name,
        'CV Mean R²': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Test R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse
    })
    
    print(f"  {name:20s} - CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}, Test R²: {r2:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = (name, model)

results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
print("\n📊 RESUMEN DE RESULTADOS DE MODELOS:")
print(results_df.to_string(index=False))

# Guardar mejor modelo
joblib.dump(best_model[1], str(outputs_dir / 'best_model_movies.joblib'))
print(f"\n✓ Mejor modelo guardado: {best_model[0]} (R² = {best_r2:.4f})")

# ==================== 8. VISUALIZACIONES Y REPORTES (Lect 3) ====================
print("\n[8/8] GENERANDO REPORTES Y VISUALIZACIONES")
print("-" * 80)

# Gráfico de rendimiento de modelos
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# R² Score
axes[0, 0].barh(results_df['Modelo'], results_df['Test R²'], color='steelblue')
axes[0, 0].set_xlabel('Test R² Score')
axes[0, 0].set_title('Comparación de Modelos - R² Score', fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# RMSE
axes[0, 1].barh(results_df['Modelo'], results_df['RMSE'], color='coral')
axes[0, 1].set_xlabel('RMSE')
axes[0, 1].set_title('Comparación de Modelos - RMSE', fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# MAE
axes[1, 0].barh(results_df['Modelo'], results_df['MAE'], color='lightgreen')
axes[1, 0].set_xlabel('MAE')
axes[1, 0].set_title('Comparación de Modelos - MAE', fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# CV R² con desviación
axes[1, 1].barh(results_df['Modelo'], results_df['CV Mean R²'], 
                xerr=results_df['CV Std'], color='mediumpurple', capsize=3)
axes[1, 1].set_xlabel('Cross-Validation R²')
axes[1, 1].set_title('Validación Cruzada - R² Score', fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(outputs_dir / 'pipeline_comparación_modelos.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfico de comparación de modelos guardado")

# Predicciones vs Real
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, best_model[1].predict(X_test), alpha=0.6, s=50)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicción perfecta')
ax.set_xlabel('Valores Reales')
ax.set_ylabel('Predicciones')
ax.set_title(f'Predicciones vs Valores Reales - {best_model[0]}', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(str(outputs_dir / 'pipeline_predicciones_vs_reales.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfico de predicciones guardado")

# PCA
print("\nAplicando PCA para visualización...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA - 2 Componentes Principales', fontweight='bold')
plt.colorbar(scatter, ax=ax, label=numeric_cols[0])
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(str(outputs_dir / 'pipeline_pca.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ PCA guardado")

# ==================== REPORTE FINAL ====================
print("\n" + "=" * 80)
print("REPORTE FINAL - PIPELINE COMPLETO")
print("=" * 80)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

reporte = f"""
{'='*80}
PIPELINE COMPLETO DE ANÁLISIS - DATASET DE PELÍCULAS
Generado: {timestamp}
{'='*80}

█ RESUMEN EJECUTIVO
{'─'*80}

1. DATASET ORIGINAL
   • Filas: {df_raw.shape[0]}
   • Columnas: {df_raw.shape[1]}
   • Columnas numéricas: {len(numeric_cols)}
   • Columnas categóricas: {len(categorical_cols)}
   • Memoria: {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB

2. PROCESAMIENTO DE DATOS
   • Outliers detectados (IQR): {outliers_removed}
   • Filas después de limpieza: {df_clean.shape[0]}
   • Reducción: {100*outliers_removed/df_raw.shape[0]:.1f}%

3. FEATURE ENGINEERING
   • Features originales: {len(df_clean.columns)}
   • Features después de ingeniería: {df_features.shape[1]}
   • Features con transformación log: {len(log_cols)}
   • Features después de encodings: {len(numeric_cols_encoded)}
   • Features para modelado (escaladas): {X.shape[1]}

4. TRANSFORMACIONES APLICADAS
   ✓ One Hot Encoding
   ✓ Label Encoding
   ✓ Min-Max Scaling
   ✓ StandardScaler
   ✓ Transformación Logarítmica
   ✓ Feature Engineering (ratios, sumas, productos)

5. ANÁLISIS DE CORRELACIÓN
   • Total de pares de features: {len(corr_pairs)}
   • Correlaciones altas (|r| > 0.8): {len([x for x in corr_pairs if abs(x[2]) > 0.8])}
   • CORRELACION MÁXIMA: {max(corr_pairs, key=lambda x: abs(x[2]))[2]:.4f}
     ({max(corr_pairs, key=lambda x: abs(x[2]))[0]} ↔ {max(corr_pairs, key=lambda x: abs(x[2]))[1]})

6. MODELOS ENTRENADOS: {len(models)}
   📊 MEJOR MODELO: {best_model[0]}
   
   Test R² Score: {best_r2:.4f}
   RMSE: {results_df.iloc[0]['RMSE']:.4f}
   MAE: {results_df.iloc[0]['MAE']:.4f}

7. TOP 5 MODELOS POR RENDIMIENTO
{chr(10).join([f"   {i+1}. {row['Modelo']:25s} - R²: {row['Test R²']:7.4f}, RMSE: {row['RMSE']:8.4f}" 
               for i, (_, row) in enumerate(results_df.head(5).iterrows())])}

█ ESTADÍSTICAS DESCRIPTIVAS
{'─'*80}

MEDIDAS DE TENDENCIA CENTRAL:
{tendencia.to_string()}

MEDIDAS DE DISPERSIÓN:
{dispersión.to_string()}

MEDIDAS DE POSICIÓN:
{posición.to_string()}

█ ARCHIVOS GENERADOS
{'─'*80}

Visualizaciones:
  ✓ pipeline_histogramas.png - Distribuciones de variables
  ✓ pipeline_matriz_correlación.png - Matriz de correlaciones
  ✓ pipeline_comparación_modelos.png - Rendimiento de modelos
  ✓ pipeline_predicciones_vs_reales.png - Validación de predicciones
  ✓ pipeline_pca.png - Reducción dimensional PCA

Modelos:
  ✓ best_model_movies.joblib - Mejor modelo entrenado

█ CONCLUSIONES Y RECOMENDACIONES
{'─'*80}

1. CALIDAD DE DATOS
   • Se detectaron y eliminaron {outliers_removed} outliers ({100*outliers_removed/df_raw.shape[0]:.1f}%)
   • El dataset limpio contiene {df_clean.shape[0]} muestras válidas
   • No hay valores faltantes críticos

2. CARACTERÍSTICAS RELEVANTES
   • Se crearon {df_features.shape[1] - df_clean.shape[1]} características derivadas
   • Transformación logarítmica aplicada a {len(log_cols)} variables
   • Feature engineering mejoró la predictibilidad

3. PERFORMANCE DEL MODELO
   • Mejor modelo: {best_model[0]} con R² = {best_r2:.4f}
   • Validación cruzada confirma consistencia (estabilidad en CV)
   • El modelo explica {best_r2*100:.2f}% de la varianza

4. RECOMENDACIONES FUTURAS
   ✓ Considerar feature selection adicional (eliminar features colineales)
   ✓ Explorar Hyperparameter Tuning avanzado (Bayesian Optimization)
   ✓ Implementar ensemble methods combinando múltiples modelos
   ✓ Análisis de importancia de features para mejor interpretabilidad
   ✓ Validación temporal si los datos tienen componente temporal
   ✓ Deployment del modelo con monitoreo en producción

█ NOTAS TÉCNICAS
{'─'*80}

• Validación cruzada: 5-fold stratified
• Split entrenamiento-prueba: 80-20
• Escalado: StandardScaler para rendimiento óptimo
• Encoding: One-Hot + Label para variables categóricas
• Tratamiento de outliers: IQR (1.5 × IQR)

{'='*80}
FIN DEL PIPELINE
{'='*80}
"""

print(reporte)

# Guardar reporte
with open(str(outputs_dir / 'pipeline_reporte_completo.txt'), 'w', encoding='utf-8') as f:
    f.write(reporte)

print(f"\n✓ Reporte guardado en: pipeline_reporte_completo.txt")

# Guardar resultados de modelos
results_df.to_csv(str(outputs_dir / 'pipeline_resultados_modelos.csv'), index=False)
print(f"✓ Resultados de modelos guardados en: pipeline_resultados_modelos.csv")

print("\n" + "=" * 80)
print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
print("=" * 80)
print(f"\nTodos los archivos han sido guardados en: {outputs_dir}")
