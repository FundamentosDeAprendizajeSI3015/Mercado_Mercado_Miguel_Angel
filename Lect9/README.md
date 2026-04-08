# Agrupamiento con K-means y DBSCAN

## Descripción

Este proyecto desarrolla un ejercicio de clustering sobre dos datasets tabulares del contexto FIRE UdeA. El script principal [Lect9/lect9.py](Lect9/lect9.py) carga los datos, aplica preprocesamiento automático para variables numéricas y categóricas, entrena modelos de agrupamiento y genera visualizaciones para analizar la estructura de los clusters.

Se trabajan dos enfoques de agrupamiento:

- `KMeans`
- `DBSCAN`

## Objetivo

El propósito del script es comparar distintas estrategias de clustering en dos datasets diferentes, evaluando:

1. el resultado de `KMeans` con un valor fijo de $K = 2$
2. la selección de un mejor valor de $K$ mediante el método del codo
3. la inercia obtenida en distintos escenarios
4. el comportamiento de `DBSCAN` en el segundo dataset

## Datasets utilizados

El análisis usa estos archivos:

- [Lect9/dataset_sintetico_FIRE_UdeA%20Lect9.csv](Lect9/dataset_sintetico_FIRE_UdeA%20Lect9.csv)
- [Lect9/dataset_sintetico_FIRE_UdeA_realista%20Lect9.csv](Lect9/dataset_sintetico_FIRE_UdeA_realista%20Lect9.csv)

Ambos datasets pueden incluir una columna `label`, pero esta se excluye del clustering para trabajar únicamente con las características.

## Flujo general del script

## 1. Configuración inicial

El script:

- define `random_state = 42` para reproducibilidad
- configura el estilo de fuente de `matplotlib`
- detecta automáticamente la carpeta base donde están los datasets
- usa la carpeta [Lect9](Lect9) como directorio de salida para las gráficas

## 2. Carga de datos

La función `cargar_dataset()`:

- lee el CSV con `pandas`
- elimina la columna `label` si existe
- devuelve el dataframe original y el conjunto de features para clustering

## 3. Preprocesamiento

La función `construir_preprocessor()` crea un `ColumnTransformer` que separa columnas numéricas y categóricas.

### Variables numéricas

Se aplica un pipeline con:

- imputación por mediana
- escalado con `StandardScaler`

### Variables categóricas

Se aplica un pipeline con:

- imputación con el valor más frecuente
- codificación con `OneHotEncoder(handle_unknown="ignore")`

Este preprocesamiento permite trabajar con datasets mixtos sin hacer limpieza manual previa.

## 4. Visualización base

La función `graficar_dataset()` usa las primeras dos variables numéricas disponibles para construir gráficos de dispersión.

Esto se utiliza para:

- visualizar el dataset original
- colorear los puntos según los clusters detectados

## 5. Análisis del Dataset 1

Para el primer dataset se realiza:

### `KMeans` con $K = 2$

- se entrena un pipeline de clustering con preprocesamiento + `KMeans`
- se reporta la inercia obtenida
- se guarda la gráfica de clusters resultante

### Método del codo

- se evalúan valores de $K$ entre 1 y 10
- se almacena la inercia de cada ajuste
- se grafica la curva de inercia para identificar el “codo”

Según el script, el valor adecuado para este dataset es $K = 2$.

## 6. Análisis del Dataset 2

Para el segundo dataset se realiza:

### `KMeans` con $K = 2$

- entrenamiento inicial con dos clusters
- cálculo de inercia
- visualización de los grupos

### Método del codo

- se prueban valores de $K$ entre 1 y 10
- se grafica la inercia
- se identifica un mejor valor de $K`

Según el análisis del script, el “codo” aparece en $K = 4$.

### `KMeans` con $K = 4$

- se reentrena el modelo usando cuatro clusters
- se calcula la nueva inercia
- se visualiza el resultado final

### `DBSCAN`

También se aplica `DBSCAN` sobre el segundo dataset con:

- `eps = 0.5`
- `min_samples = 5`

Después se grafica el resultado del agrupamiento y se inspeccionan las etiquetas generadas.

## Dependencias

Instala las librerías necesarias con:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Ejecución

Desde la raíz del proyecto:

```bash
python Lect9/lect9.py
```

O desde la carpeta [Lect9](Lect9):

```bash
python lect9.py
```

## Archivos generados

Las gráficas se guardan en la carpeta [Lect9](Lect9).

### Dataset 1

- [Lect9/dataset_1_variables_numericas.png](Lect9/dataset_1_variables_numericas.png)
- [Lect9/dataset_1_kmeans_k2.png](Lect9/dataset_1_kmeans_k2.png)
- [Lect9/dataset_1_metodo_codo.png](Lect9/dataset_1_metodo_codo.png)

### Dataset 2

- [Lect9/dataset_2_variables_numericas.png](Lect9/dataset_2_variables_numericas.png)
- [Lect9/dataset_2_kmeans_k2.png](Lect9/dataset_2_kmeans_k2.png)
- [Lect9/dataset_2_metodo_codo.png](Lect9/dataset_2_metodo_codo.png)
- [Lect9/dataset_2_kmeans_k4.png](Lect9/dataset_2_kmeans_k4.png)
- [Lect9/dataset_2_dbscan.png](Lect9/dataset_2_dbscan.png)

## Conceptos cubiertos

Este ejercicio permite practicar:

- aprendizaje no supervisado
- clustering con `KMeans`
- clustering basado en densidad con `DBSCAN`
- método del codo
- análisis de inercia
- preprocesamiento mixto con `ColumnTransformer`
- imputación de valores faltantes
- escalado de variables numéricas
- codificación one-hot de variables categóricas
- visualización de clusters en 2D

## Observaciones importantes

- Las gráficas usan únicamente las dos primeras columnas numéricas para visualización.
- El clustering realmente se entrena sobre todas las variables disponibles después del preprocesamiento.
- Si un dataset no tiene al menos dos columnas numéricas, el script lanzará un error al intentar graficar.
- La columna `label`, si existe, se elimina antes del agrupamiento.
- La interpretación del “mejor” valor de $K$ se hace visualmente a partir del método del codo.

## Posibles mejoras

1. calcular métricas adicionales como `silhouette_score`
2. automatizar la selección óptima de $K$
3. comparar con otros algoritmos como `AgglomerativeClustering`
4. añadir reducción de dimensionalidad con PCA o UMAP antes de graficar
5. guardar un reporte con las inercias y número de clusters detectados
6. ajustar `DBSCAN` con búsqueda sistemática de `eps` y `min_samples`

## Archivo relacionado

También existe un notebook complementario en [Lect9/ejAgrupamiento_kmeans_dbscan.ipynb](Lect9/ejAgrupamiento_kmeans_dbscan.ipynb).

## Autor

Miguel Angel Mercado