import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import uniform, randint
import warnings
import os
warnings.filterwarnings('ignore')

# Crear directorio de outputs - usando ruta absoluta basada en el directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'outputs')
os.makedirs(output_dir, exist_ok=True)
print(f"Directorio de outputs: {output_dir}\n")

print("=" * 80)
print("1. DESCARGANDO Y CARGANDO EL DATASET")
print("=" * 80)

path = kagglehub.dataset_download("bharatnatrayn/movies-dataset-for-feature-extracion-prediction")
print(f"Path to dataset files: {path}\n")

# Listar archivos disponibles
import os
files = os.listdir(path)
print(f"Archivos disponibles: {files}\n")

# Encontrar el archivo CSV
csv_file = None
for file in files:
    if file.endswith('.csv'):
        csv_file = file
        break

if csv_file is None:
    raise FileNotFoundError(f"No se encontró ningún archivo CSV en {path}")

# Cargar el dataset
df = pd.read_csv(f"{path}/{csv_file}")
print(f"Dataset cargado desde: {csv_file}\n")
print(f"Dataset shape: {df.shape}")
print(f"\nPrimeras filas:\n{df.head()}")
print(f"\nTipos de datos:\n{df.dtypes}")
print(f"\nValores faltantes:\n{df.isnull().sum()}")
print(f"\nEstadísticas descriptivas:\n{df.describe()}")

print("\n" + "=" * 80)
print("2. EXPLORACIÓN GRÁFICA Y LIMPIEZA DEL DATASET")
print("=" * 80)

# Visualizar distribuciones
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
for idx, col in enumerate(numeric_cols):
    ax = axes[idx // 2, idx % 2]
    ax.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(f'Distribución de {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frecuencia')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_distribucion_variables.png'), dpi=300, bbox_inches='tight')
plt.close()

# Matriz de correlación
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Matriz de Correlación')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_matriz_correlacion.png'), dpi=300, bbox_inches='tight')
plt.close()

# Limpieza y transformación
print(f"\nFilas originales: {len(df)}")
# Eliminar duplicados
df = df.drop_duplicates()
print(f"Después de eliminar duplicados: {len(df)}")

# Manejo de valores faltantes - MEJORADO
# Primero, eliminar filas donde faltan valores críticos
df = df.dropna(subset=['RATING', 'RunTime'])
print(f"Después de eliminar filas sin RATING o RunTime: {len(df)}")

# Para otras columnas, imputar antes de codificar
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna('Unknown', inplace=True)

print(f"Datos después de limpieza: {len(df)} filas")

# Codificar variables categóricas
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"Codificada columna: {col}")

print("\n" + "=" * 80)
print("1.0.1 REGRESIÓN LINEAL (RIDGE Y LASSO)")
print("=" * 80)

# Determinar columna objetivo para regresión lineal
# Seleccionamos una columna numérica REAL como objetivo (RATING)
target_regression = 'RATING'
print(f"\nColumna seleccionada para regresión lineal: {target_regression}")

X_reg = df.drop(columns=[target_regression])
y_reg = df[target_regression]

# Verificar que no haya NaN en los datos
print(f"NaN en X: {X_reg.isnull().sum().sum()}")
print(f"NaN en y: {y_reg.isnull().sum()}")

# Dividir en entrenamiento y prueba
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
print(f"Entrenamiento: {X_train_reg.shape[0]} muestras")
print(f"Prueba: {X_test_reg.shape[0]} muestras")

# Graficar entrenamiento y prueba (primeras 100 muestras para claridad)
plt.figure(figsize=(12, 6))
plt.scatter(range(100), y_train_reg.values[:100], color='blue', alpha=0.6, label='Entrenamiento', s=50)
plt.scatter(range(100), y_test_reg.values[:100], color='red', alpha=0.6, label='Prueba', s=50)
plt.xlabel('Índice de muestra')
plt.ylabel(target_regression)
plt.title(f'Conjunto de Entrenamiento vs Prueba ({target_regression})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_train_test_split.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfico de entrenamiento/prueba guardado")

# Definir pipelines con Ridge y Lasso
pipeline_ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

pipeline_lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso())
])

# Definir distribuciones de parámetros para búsqueda aleatoria
param_dist_ridge = {
    'ridge__alpha': uniform(0.001, 100),
}

param_dist_lasso = {
    'lasso__alpha': uniform(0.001, 100),
}

print("\n--- Búsqueda Aleatoria y Cross-Validation para Ridge ---")
random_search_ridge = RandomizedSearchCV(
    pipeline_ridge, 
    param_dist_ridge, 
    n_iter=50, 
    cv=5, 
    random_state=42,
    n_jobs=-1,
    verbose=1
)
random_search_ridge.fit(X_train_reg, y_train_reg)

print(f"\nMejores parámetros Ridge: {random_search_ridge.best_params_}")
print(f"Mejor score Ridge (CV): {random_search_ridge.best_score_:.4f}")

print("\n--- Búsqueda Aleatoria y Cross-Validation para Lasso ---")
random_search_lasso = RandomizedSearchCV(
    pipeline_lasso, 
    param_dist_lasso, 
    n_iter=50, 
    cv=5, 
    random_state=42,
    n_jobs=-1,
    verbose=1
)
random_search_lasso.fit(X_train_reg, y_train_reg)

print(f"\nMejores parámetros Lasso: {random_search_lasso.best_params_}")
print(f"Mejor score Lasso (CV): {random_search_lasso.best_score_:.4f}")

# Predicciones
y_pred_ridge = random_search_ridge.predict(X_test_reg)
y_pred_lasso = random_search_lasso.predict(X_test_reg)

# Métricas
r2_ridge = r2_score(y_test_reg, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test_reg, y_pred_ridge)

r2_lasso = r2_score(y_test_reg, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test_reg, y_pred_lasso)

print("\n" + "=" * 60)
print("RESULTADOS REGRESIÓN LINEAL")
print("=" * 60)
print(f"Ridge - R²: {r2_ridge:.4f}, MAE: {mae_ridge:.4f}")
print(f"Lasso - R²: {r2_lasso:.4f}, MAE: {mae_lasso:.4f}")

# Graficar predicciones
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ridge
axes[0].scatter(y_test_reg, y_pred_ridge, alpha=0.6, color='blue')
axes[0].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
axes[0].set_xlabel('Valores Reales')
axes[0].set_ylabel('Valores Predichos')
axes[0].set_title(f'Ridge - R²: {r2_ridge:.4f}, MAE: {mae_ridge:.4f}')
axes[0].grid(True, alpha=0.3)

# Lasso
axes[1].scatter(y_test_reg, y_pred_lasso, alpha=0.6, color='green')
axes[1].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
axes[1].set_xlabel('Valores Reales')
axes[1].set_ylabel('Valores Predichos')
axes[1].set_title(f'Lasso - R²: {r2_lasso:.4f}, MAE: {mae_lasso:.4f}')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_regresion_lineal_predicciones.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfico de predicciones guardado")

print("\n" + "=" * 80)
print("1.0.2 REGRESIÓN LOGÍSTICA")
print("=" * 80)

# Crear variable binaria basada en RATING (Good vs Not Good)
target_classification = "rating_good"
df[target_classification] = (df['RATING'] >= 7.0).astype(int)

print(f"\nColumna seleccionada para regresión logística: {target_classification}")
print(f"Definición: Rating >= 7.0 = 'Good Movie' (1), Rating < 7.0 = 'Not Good' (0)")
print(f"Distribución de clases:\n{df[target_classification].value_counts()}")

# Verificar balance
class_counts = df[target_classification].value_counts()
if len(class_counts) < 2:
    print("ERROR: No hay dos clases. Creando variable alternativa...")
    # Si no funciona, usar otro método
    df[target_classification] = (df['RunTime'] > df['RunTime'].median()).astype(int)
    print(f"Usando: RunTime > mediana")
    print(f"Distribución de clases:\n{df[target_classification].value_counts()}")

X_clf = df.drop(columns=[target_classification])
y_clf = df[target_classification]

# Dividir en entrenamiento y prueba
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
print(f"\nEntrenamiento: {X_train_clf.shape[0]} muestras")
print(f"Prueba: {X_test_clf.shape[0]} muestras")
print(f"Clases en entrenamiento: {y_train_clf.value_counts().to_dict()}")

# Definir pipeline
pipeline_logistic = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'))
])

# Definir distribuciones de parámetros - SIMPLIFICADO
param_dist_logistic = {
    'logistic__C': uniform(0.01, 10),
}

print("\n--- Búsqueda Aleatoria y Cross-Validation para Regresión Logística ---")
random_search_logistic = RandomizedSearchCV(
    pipeline_logistic,
    param_dist_logistic,
    n_iter=30,
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    scoring='f1'
)
random_search_logistic.fit(X_train_clf, y_train_clf)

print(f"\nMejores parámetros: {random_search_logistic.best_params_}")
print(f"Mejor score (CV-F1): {random_search_logistic.best_score_:.4f}")

# Predicciones
y_pred_clf = random_search_logistic.predict(X_test_clf)
y_pred_proba = random_search_logistic.predict_proba(X_test_clf)[:, 1]

# Métricas
accuracy = accuracy_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)

print("\n" + "=" * 60)
print("RESULTADOS REGRESIÓN LOGÍSTICA")
print("=" * 60)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")

# Matriz de confusión
cm = confusion_matrix(y_test_clf, y_pred_clf)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negativo', 'Positivo'])
disp.plot(ax=ax, cmap='Blues')
plt.title(f'Matriz de Confusión - Accuracy: {accuracy:.4f}')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '05_matriz_confusion.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Matriz de confusión guardada")

# Graficar ROC-like (predicciones vs probabilidades)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma de probabilidades predichas
axes[0].hist(y_pred_proba[y_test_clf == 0], bins=30, alpha=0.6, label='Clase 0', color='blue')
axes[0].hist(y_pred_proba[y_test_clf == 1], bins=30, alpha=0.6, label='Clase 1', color='red')
axes[0].set_xlabel('Probabilidad predicha (Clase 1)')
axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Distribución de Probabilidades Predichas')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Predicciones vs Valores reales
axes[1].scatter(range(len(y_test_clf)), y_test_clf, alpha=0.6, label='Valores Reales', color='blue', s=50)
axes[1].scatter(range(len(y_test_clf)), y_pred_clf, alpha=0.6, label='Predicciones', color='red', marker='x', s=100)
axes[1].set_xlabel('Índice de muestra')
axes[1].set_ylabel('Clase')
axes[1].set_title(f'Predicciones vs Valores Reales (Accuracy: {accuracy:.4f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '06_regresion_logistica_predicciones.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfico de predicciones guardado")

print("\n" + "=" * 80)
print("RESUMEN FINAL")
print("=" * 80)
print(f"\n✓ Análisis completado exitosamente")
print(f"✓ {len(df)} muestras analizadas")
print(f"\nArchivos generados:")
print(f"  1. Distribución de variables")
print(f"  2. Matriz de correlación")
print(f"  3. Train/Test split visualization")
print(f"  4. Predicciones Ridge y Lasso")
print(f"  5. Matriz de confusión")
print(f"  6. Predicciones Regresión Logística")