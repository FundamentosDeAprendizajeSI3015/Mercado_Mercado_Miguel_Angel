import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Crear ruta absoluta para outputs
script_dir = Path(__file__).parent.parent
outputs_dir = script_dir / 'outputs'
outputs_dir.mkdir(exist_ok=True)

# =============================================================================
# 1. DESCARGAR Y CARGAR EL DATASET
# =============================================================================
print("=" * 80)
print("1. DESCARGANDO Y CARGANDO EL DATASET")
print("=" * 80)

path = kagglehub.dataset_download("bharatnatrayn/movies-dataset-for-feature-extracion-prediction")
print(f"Path to dataset files: {path}\n")

# Cargar el dataset
df = pd.read_csv(f"{path}/movies.csv")
print(f"Dataset cargado: {df.shape[0]} filas y {df.shape[1]} columnas\n")
print("Primeras filas del dataset:")
print(df.head())
print("\nInformación del dataset:")
print(df.info())
print("\nEstadísticas básicas:")
print(df.describe())

# =============================================================================
# 2. MEDIDAS DE TENDENCIA CENTRAL
# =============================================================================
print("\n" + "=" * 80)
print("2. MEDIDAS DE TENDENCIA CENTRAL")
print("=" * 80)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nColumnas numéricas: {numeric_cols}\n")

tendencia_central = pd.DataFrame({
    'Media (Mean)': df[numeric_cols].mean(),
    'Mediana (Median)': df[numeric_cols].median(),
    'Moda (Mode)': [df[col].mode()[0] if len(df[col].mode()) > 0 else np.nan for col in numeric_cols]
})

print("Medidas de Tendencia Central:")
print(tendencia_central)

# =============================================================================
# 3. MEDIDAS DE DISPERSIÓN
# =============================================================================
print("\n" + "=" * 80)
print("3. MEDIDAS DE DISPERSIÓN")
print("=" * 80)

dispersión = pd.DataFrame({
    'Desviación Estándar': df[numeric_cols].std(),
    'Varianza': df[numeric_cols].var(),
    'Rango': df[numeric_cols].max() - df[numeric_cols].min(),
    'Coeficiente de Variación': (df[numeric_cols].std() / df[numeric_cols].mean()) * 100
})

print("\nMedidas de Dispersión:")
print(dispersión)

# =============================================================================
# 4. MEDIDAS DE POSICIÓN Y DETECCIÓN DE OUTLIERS
# =============================================================================
print("\n" + "=" * 80)
print("4. MEDIDAS DE POSICIÓN Y DETECCIÓN DE OUTLIERS")
print("=" * 80)

medidas_posición = pd.DataFrame({
    'Q1 (25%)': df[numeric_cols].quantile(0.25),
    'Q2 (50%)': df[numeric_cols].quantile(0.50),
    'Q3 (75%)': df[numeric_cols].quantile(0.75),
    'IQR': df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
})

print("\nMedidas de Posición (Cuartiles):")
print(medidas_posición)

# Detectar outliers usando IQR
print("\n--- Detección de Outliers (Método IQR) ---")
outliers_count = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outliers_count[col] = len(outliers)
    print(f"{col}: {len(outliers)} outliers detectados [{lower_bound:.2f}, {upper_bound:.2f}]")

# Eliminar outliers
df_clean = df.copy()
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

print(f"\nDataset después de eliminar outliers: {df_clean.shape[0]} filas (se eliminaron {df.shape[0] - df_clean.shape[0]} filas)")

# =============================================================================
# 5. HISTOGRAMAS - ANÁLISIS DE DISTRIBUCIONES
# =============================================================================
print("\n" + "=" * 80)
print("5. GENERANDO HISTOGRAMAS")
print("=" * 80)

# Crear histogramas para columnas numéricas
n_cols = len(numeric_cols)
n_rows = (n_cols + 3) // 4
fig, axes = plt.subplots(n_rows, 4, figsize=(18, n_rows * 4))
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    axes[idx].hist(df_clean[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Histograma de {col}', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frecuencia')
    axes[idx].grid(alpha=0.3)
    
    # Calcular asimetría y curtosis
    skewness = stats.skew(df_clean[col])
    kurtosis = stats.kurtosis(df_clean[col])
    axes[idx].text(0.98, 0.97, f'Asimetría: {skewness:.3f}\nCurtosis: {kurtosis:.3f}', 
                   transform=axes[idx].transAxes, fontsize=8, verticalalignment='top', 
                   horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Eliminar subplots vacíos
for idx in range(n_cols, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(str(outputs_dir / 'histogramas_distribuciones.png'), dpi=300, bbox_inches='tight')
print(f"Histogramas guardados en: {outputs_dir / 'histogramas_distribuciones.png'}")
plt.close()

# =============================================================================
# 6. GRÁFICOS DE DISPERSIÓN - ANÁLISIS DE RELACIONES
# =============================================================================
print("\n" + "=" * 80)
print("6. GENERANDO GRÁFICOS DE DISPERSIÓN - ANÁLISIS DE RELACIONES")
print("=" * 80)

# Seleccionar pares de columnas numéricas más interesantes
if len(numeric_cols) >= 2:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Crear gráficos de dispersión entre pares de columnas
    pairs = [
        (numeric_cols[0], numeric_cols[1]),
        (numeric_cols[0], numeric_cols[2]) if len(numeric_cols) > 2 else (numeric_cols[0], numeric_cols[1]),
        (numeric_cols[1], numeric_cols[2]) if len(numeric_cols) > 2 else (numeric_cols[0], numeric_cols[1]),
        (numeric_cols[-1], numeric_cols[-2])
    ]
    
    for idx, (col1, col2) in enumerate(pairs[:4]):
        axes[idx].scatter(df_clean[col1], df_clean[col2], alpha=0.5, s=30)
        # Calcular correlación
        corr = df_clean[col1].corr(df_clean[col2])
        axes[idx].set_xlabel(col1, fontweight='bold')
        axes[idx].set_ylabel(col2, fontweight='bold')
        axes[idx].set_title(f'Relación entre {col1} y {col2}\n(Correlación: {corr:.3f})', fontweight='bold')
        axes[idx].grid(alpha=0.3)
        
        # Agregar línea de tendencia
        z = np.polyfit(df_clean[col1].dropna(), df_clean[col2].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_clean[col1].min(), df_clean[col1].max(), 100)
        axes[idx].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(str(outputs_dir / 'graficos_dispersión.png'), dpi=300, bbox_inches='tight')
    print("Gráficos de dispersión guardados en: " + str(outputs_dir / 'graficos_dispersión.png'))
    plt.close()

# =============================================================================
# 7. TRANSFORMACIONES DE COLUMNAS
# =============================================================================
print("\n" + "=" * 80)
print("7. TRANSFORMACIONES DE COLUMNAS")
print("=" * 80)

# Copiar dataset para transformaciones
df_transformed = df_clean.copy()

# Identificar columnas categóricas
categorical_cols = df_transformed.select_dtypes(include=['object']).columns.tolist()
print(f"\nColumnas categóricas: {categorical_cols}")

# ONE HOT ENCODING
print("\n--- One Hot Encoding ---")
if len(categorical_cols) > 0:
    for col in categorical_cols:
        if df_transformed[col].nunique() <= 10:  # Solo para columnas con pocos valores únicos
            one_hot = pd.get_dummies(df_transformed[col], prefix=col, drop_first=False)
            df_transformed = pd.concat([df_transformed, one_hot], axis=1)
            print(f"One Hot Encoding aplicado a '{col}': {one_hot.shape[1]} columnas creadas")

# LABEL ENCODING
print("\n--- Label Encoding ---")
label_encoders = {}
for col in categorical_cols:
    if col in df_transformed.columns and df_transformed[col].dtype == 'object':
        le = LabelEncoder()
        df_transformed[f'{col}_encoded'] = le.fit_transform(df_transformed[col].fillna('Unknown'))
        label_encoders[col] = le
        print(f"Label Encoding aplicado a '{col}'")

print(f"\nDataset después de transformaciones: {df_transformed.shape[0]} filas y {df_transformed.shape[1]} columnas")

# =============================================================================
# 8. CORRELACIÓN DE COLUMNAS
# =============================================================================
print("\n" + "=" * 80)
print("8. ANÁLISIS DE CORRELACIÓN")
print("=" * 80)

numeric_cols_transformed = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = df_transformed[numeric_cols_transformed].corr()

print(f"\nColumnas numéricas después de transformaciones: {len(numeric_cols_transformed)}")
print("\nMatriz de Correlación:")
print(correlation_matrix)

# Crear mapa de calor de correlaciones
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
plt.title('Matriz de Correlación - Dataset de Películas', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(str(outputs_dir / 'matriz_correlación.png'), dpi=300, bbox_inches='tight')
print("\nMatriz de correlación guardada en: " + str(outputs_dir / 'matriz_correlación.png'))
plt.close()

# Identificar columnas altamente correlacionadas
print("\n--- Columnas altamente correlacionadas (|corr| > 0.8) excluyendo diagonal ---")
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            corr_val = correlation_matrix.iloc[i, j]
            high_corr_pairs.append((col1, col2, corr_val))
            print(f"{col1} ↔ {col2}: {corr_val:.3f}")

if len(high_corr_pairs) == 0:
    print("No hay columnas altamente correlacionadas")

# =============================================================================
# 9. ESCALADO - MIN-MAX SCALING Y STANDARDSCALER
# =============================================================================
print("\n" + "=" * 80)
print("9. ESCALADO DE COLUMNAS")
print("=" * 80)

df_scaled_minmax = df_transformed[numeric_cols_transformed].copy()
df_scaled_standard = df_transformed[numeric_cols_transformed].copy()

# Min-Max Scaling
minmax_scaler = MinMaxScaler()
df_scaled_minmax = pd.DataFrame(
    minmax_scaler.fit_transform(df_scaled_minmax),
    columns=[f'{col}_minmax' for col in numeric_cols_transformed]
)

# StandardScaler
standard_scaler = StandardScaler()
df_scaled_standard = pd.DataFrame(
    standard_scaler.fit_transform(df_scaled_standard),
    columns=[f'{col}_standard' for col in numeric_cols_transformed]
)

print(f"Min-Max Scaling aplicado: {df_scaled_minmax.shape[1]} columnas creadas")
print(f"StandardScaler aplicado: {df_scaled_standard.shape[1]} columnas creadas")

print("\nEjemplo de escalado (primeras 5 filas):")
print("\nOriginal (primeras 3 columnas numéricas):")
print(df_transformed[numeric_cols_transformed[:3]].head())
print("\nMin-Max Scaled (primeras 3 columnas):")
print(df_scaled_minmax[[f'{col}_minmax' for col in numeric_cols_transformed[:3]]].head())
print("\nStandardScaled (primeras 3 columnas):")
print(df_scaled_standard[[f'{col}_standard' for col in numeric_cols_transformed[:3]]].head())

# =============================================================================
# 10. TRANSFORMACIÓN LOGARÍTMICA
# =============================================================================
print("\n" + "=" * 80)
print("10. TRANSFORMACIÓN LOGARÍTMICA")
print("=" * 80)

df_log_transformed = df_transformed.copy()
log_applied = []

for col in numeric_cols_transformed:
    if (df_transformed[col] > 0).all():
        df_log_transformed[f'{col}_log'] = np.log(df_transformed[col])
        log_applied.append(col)
        
        # Visualizar transformación
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df_transformed[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_title(f'Original: {col}', fontweight='bold')
        axes[0].set_xlabel('Valor')
        axes[0].set_ylabel('Frecuencia')
        axes[0].grid(alpha=0.3)
        
        axes[1].hist(df_log_transformed[f'{col}_log'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1].set_title(f'Log Transformada: {col}', fontweight='bold')
        axes[1].set_xlabel('Log(Valor)')
        axes[1].set_ylabel('Frecuencia')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(str(outputs_dir / f'transformacion_log_{col}.png'), dpi=300, bbox_inches='tight')
        plt.close()

print(f"\nTransformación logarítmica aplicada a {len(log_applied)} columnas:")
for col in log_applied:
    print(f"  - {col}")

# =============================================================================
# 11. CONCLUSIONES
# =============================================================================
print("\n" + "=" * 80)
print("CONCLUSIONES Y HALLAZGOS PRINCIPALES")
print("=" * 80)

conclusions = """
RESUMEN DE ANÁLISIS DEL DATASET DE PELÍCULAS

█ ESTRUCTURA DEL DATASET:
  • Total de filas originales: {}
  • Total de filas después de limpiar (sin outliers): {}
  • Total de columnas numéricas: {}
  • Total de columnas categóricas: {}
  • Filas eliminadas por outliers: {} ({:.1f}%)

█ MEDIDAS DE TENDENCIA CENTRAL:
  • Se calcularon Media, Mediana y Moda para todas las columnas numéricas
  • La variabilidad entre estas medidas indica la simetría de las distribuciones
  • Distribuciones asimétricas pueden beneficiarse de transformaciones logarítmicas

█ MEDIDAS DE DISPERSIÓN:
  • Se identificó la variabilidad de cada columna mediante desviación estándar y varianza
  • El coeficiente de variación permite comparar la dispersión relativa entre columnas
  • Columnas con alto CV pueden requerir normalización

█ DETECCIÓN Y TRATAMIENTO DE OUTLIERS:
  • Se utilizó el método de Rango Intercuartílico (IQR) para identificar outliers
  • Se eliminaron {} outliers del dataset
  • Los outliers pueden representar casos anormales o errores de medición

█ DISTRIBUCIONES (HISTOGRAMAS):
  • Se generaron histogramas para visualizar la forma de cada distribución
  • Análisis de Asimetría (Skewness): indica sesgo hacia izquierda/derecha
  • Análisis de Curtosis: indica concentración en las colas vs. pico central
  • La transformación logarítmica se aplicó a {} columnas con distribuciones sesgadas

█ RELACIONES ENTRE VARIABLES (GRÁFICOS DE DISPERSIÓN):
  • Se analizaron {} pares de variables para identificar relaciones lineales
  • Se calcularon coeficientes de correlación entre variables
  • Se trazaron líneas de tendencia para visualizar relaciones

█ CORRELACIÓN ENTRE COLUMNAS:
  • Total de columnas analizadas: {}
  • Pares altamente correlacionados (|r| > 0.8): {}
  • Las columnas altamente correlacionadas pueden ser redundantes
  • Considera eliminar una de cada par con alta correlación

█ TRANSFORMACIONES REALIZADAS:
  • One Hot Encoding: Conversión de variables categóricas a binarias
  • Label Encoding: Asignación de números a categorías
  • Min-Max Scaling: Escalado de valores al rango [0, 1]
  • StandardScaler: Normalización con media 0 y desviación estándar 1
  • Transformación Logarítmica: Aplicada a {} columnas para mejorar simetría

█ RECOMENDACIONES:
  1. Usar StandardScaler para algoritmos como KNN, SVM y Regresión Lineal
  2. Usar Min-Max Scaling para redes neuronales y algoritmos basados en distancia
  3. Eliminar columnas con correlación > 0.95 para evitar multicolinealidad
  4. Para datos categóricos con muchas categorías, considerar target encoding
  5. Explorar interacciones entre variables importantes
  6. Considerar feature engineering para mejorar predictibilidad

█ PRÓXIMOS PASOS:
  1. Seleccionar características relevantes basadas en análisis de correlación
  2. Dividir datos en entrenamiento y prueba
  3. Aplicar transformaciones elegidas consistentemente a ambos conjuntos
  4. Validar normalidad de distribuciones post-transformación
  5. Entrenar modelos con diferentes combinaciones de features
""".format(
    df.shape[0],
    df_clean.shape[0],
    len(numeric_cols),
    len(categorical_cols),
    df.shape[0] - df_clean.shape[0],
    ((df.shape[0] - df_clean.shape[0]) / df.shape[0]) * 100,
    df.shape[0] - df_clean.shape[0],
    len(log_applied),
    4,
    len(numeric_cols_transformed),
    len(high_corr_pairs),
    len(log_applied)
)

print(conclusions)

# Guardar conclusiones en archivo
with open(str(outputs_dir / 'analisis_conclusiones.txt'), 'w') as f:
    f.write(conclusions)

print("\nArchivo de conclusiones guardado en: " + str(outputs_dir / 'analisis_conclusiones.txt'))
print("\n" + "=" * 80)
print("ANÁLISIS COMPLETADO")
print("=" * 80)