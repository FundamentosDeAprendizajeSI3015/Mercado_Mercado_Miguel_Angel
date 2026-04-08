# Clasificación supervisada con Random Forest y Gradient Boosting sobre un dataset de películas

## Descripción

Este proyecto desarrolla un flujo de clasificación supervisada a partir de un dataset de películas descargado desde Kaggle. El script [Lect6/lect6.py](Lect6/lect6.py) automatiza la carga del dataset, la detección de una variable objetivo, la limpieza de datos, el preprocesamiento mixto para variables numéricas y categóricas, el entrenamiento de modelos ensemble y la exportación de métricas, gráficas y modelos entrenados.

El análisis compara principalmente dos algoritmos:

- `RandomForestClassifier`
- `GradientBoostingClassifier`

## Objetivo

El propósito del script es construir un pipeline reutilizable para clasificación binaria, capaz de:

1. descargar y localizar automáticamente el archivo del dataset
2. identificar una columna objetivo razonable
3. transformar una variable continua o categórica en una variable `target`
4. limpiar columnas poco útiles o con demasiados nulos
5. preprocesar datos numéricos y categóricos
6. entrenar modelos con búsqueda aleatoria de hiperparámetros
7. evaluar el rendimiento con métricas de clasificación
8. guardar resultados, gráficas y modelos en disco

## Dataset utilizado

El script descarga automáticamente el dataset:

- **Fuente:** Kaggle
- **Identificador:** `bharatnatrayn/movies-dataset-for-feature-extracion-prediction`
- **Entrada esperada:** uno o más archivos CSV dentro del recurso descargado

## Flujo general del script

### 1. Descarga y carga de datos

Se usa `kagglehub` para descargar el dataset. El script soporta distintos escenarios:

- ruta a directorio
- ruta a archivo
- archivo `.zip`, que puede extraerse automáticamente

Después, busca el primer archivo CSV disponible y lo carga con `pandas`.

### 2. Exploración rápida

Se imprime una vista inicial del dataset y se guarda un resumen en [outputs/dataset_quickinfo.txt](outputs/dataset_quickinfo.txt), incluyendo:

- `info()` del dataset
- conteo de valores nulos
- descripción estadística general

### 3. Detección automática de la variable objetivo

El script intenta encontrar una columna objetivo siguiendo este orden de prioridad:

- `rating`
- `votes`
- `genre`
- alguna columna numérica que contenga palabras como `revenue`, `profit` o `score`

Si la variable detectada es numérica, se transforma en una variable binaria `target` usando un umbral.

- Para `rating`, se usa preferiblemente $7.0$
- En otros casos, se usa la mediana

Si la variable es categórica, se factoriza numéricamente.

### 4. Limpieza de columnas

Se eliminan columnas que pueden perjudicar el modelado:

- texto muy largo
- columnas con cardinalidad excesiva
- columnas con más de 60% de valores nulos
- identificadores evidentes como `id`, `movie_id`, `imdb_id`, `title`, `overview`, etc.

### 5. Preparación de variables predictoras

Se construyen:

- `y`: columna `target`
- `X`: todas las demás columnas útiles, excluyendo `target` y la columna objetivo original

Luego se separan las columnas en:

- numéricas
- categóricas

### 6. Preprocesamiento

El script usa `ColumnTransformer` con dos pipelines:

#### Variables numéricas
- imputación por mediana
- escalado con `StandardScaler`

#### Variables categóricas
- imputación con valor constante `missing`
- codificación con `OneHotEncoder`

### 7. Entrenamiento de modelos

Se usa `train_test_split` con estratificación para clasificación binaria.

Posteriormente se entrenan dos modelos:

#### `RandomForestClassifier`
Búsqueda aleatoria sobre:
- `n_estimators`
- `max_depth`

#### `GradientBoostingClassifier`
Búsqueda aleatoria sobre:
- `n_estimators`
- `learning_rate`
- `max_depth`

Ambos modelos se integran en un `Pipeline` junto con el preprocesador.

### 8. Evaluación

Para cada modelo se calculan las siguientes métricas:

- `accuracy`
- `precision`
- `recall`
- `f1`

Además se genera una matriz de confusión y, cuando el modelo lo permite, una visualización de importancia de características.

### 9. Exportación de resultados

Se guardan:

- modelos entrenados en formato `.joblib`
- matrices de confusión
- gráficas de importancia de características
- archivo comparativo de métricas en CSV

## Dependencias

Instala las librerías necesarias con:

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn scipy joblib
```

## Ejecución

Desde la raíz del proyecto:

```bash
python Lect6/lect6.py
```

O desde la carpeta [Lect6](Lect6):

```bash
python lect6.py
```

## Archivos generados

Los resultados se guardan en la carpeta [outputs](outputs).

### Archivos principales

- [outputs/dataset_quickinfo.txt](outputs/dataset_quickinfo.txt)
- [outputs/confusion_matrix_random_forest.png](outputs/confusion_matrix_random_forest.png)
- [outputs/confusion_matrix_gradient_boosting.png](outputs/confusion_matrix_gradient_boosting.png)
- [outputs/feature_importances_random_forest.png](outputs/feature_importances_random_forest.png)
- [outputs/feature_importances_gradient_boosting.png](outputs/feature_importances_gradient_boosting.png)
- [outputs/random_forest_model.joblib](outputs/random_forest_model.joblib)
- [outputs/gradient_boosting_model.joblib](outputs/gradient_boosting_model.joblib)
- [outputs/metrics_comparison.csv](outputs/metrics_comparison.csv)

> Nota: los archivos de importancia de características solo se generan si el modelo expone `feature_importances_` y si el proceso de reconstrucción de nombres de variables funciona correctamente.

## Métricas utilizadas

### `Accuracy`
Proporción de predicciones correctas sobre el total.

### `Precision`
Mide cuántos positivos predichos realmente eran positivos.

### `Recall`
Mide cuántos positivos reales fueron detectados por el modelo.

### `F1`
Media armónica entre `precision` y `recall`:

$$
F1 = 2 \cdot \frac{precision \cdot recall}{precision + recall}
$$

## Conceptos cubiertos

Este ejercicio permite practicar:

- clasificación supervisada
- construcción de pipelines en `scikit-learn`
- imputación de faltantes
- codificación one-hot
- escalado de variables
- búsqueda aleatoria de hiperparámetros
- validación por partición entrenamiento/prueba
- matrices de confusión
- exportación de modelos entrenados
- interpretación básica con importancia de variables

## Observaciones importantes

- La detección de la columna objetivo es automática, por lo que puede requerir ajustes manuales si el dataset cambia.
- La binarización del objetivo puede modificar el significado original del problema.
- El script elimina columnas de texto largo y alta cardinalidad para simplificar el modelado.
- La salida se guarda en la carpeta global [outputs](outputs), no dentro de [Lect6](Lect6).
- `GradientBoostingClassifier` no siempre maneja igual de bien datasets muy grandes o altamente desbalanceados sin ajustes adicionales.

## Posibles mejoras

1. permitir al usuario definir manualmente la columna objetivo
2. agregar métricas como ROC-AUC
3. incorporar validación cruzada estratificada más robusta
4. manejar desbalanceo con `class_weight` o técnicas de resampling
5. registrar los mejores hiperparámetros en un reporte de texto
6. agregar explicabilidad con SHAP o permutation importance
7. validar automáticamente si la tarea resultante es binaria o multiclase

## Autor

Miguel Angel Mercado