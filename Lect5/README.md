# Modelado supervisado con regresión lineal y logística sobre un dataset de películas

## Descripción

Este proyecto implementa un flujo completo de análisis y modelado supervisado usando un dataset de películas descargado desde Kaggle. El script [Lect5/lect5.py](Lect5/lect5.py) realiza limpieza de datos, codificación de variables categóricas, análisis gráfico, entrenamiento de modelos de regresión y clasificación, y generación de visualizaciones de resultados.

El enfoque principal es comparar modelos de:

- `Ridge`
- `Lasso`
- `LogisticRegression`

además de aplicar búsqueda aleatoria de hiperparámetros con validación cruzada.

## Dataset utilizado

El script descarga automáticamente el dataset:

- **Fuente:** Kaggle
- **Identificador:** `bharatnatrayn/movies-dataset-for-feature-extracion-prediction`
- **Formato esperado:** archivo CSV dentro de la carpeta descargada

## Objetivos del script

El archivo [Lect5/lect5.py](Lect5/lect5.py) tiene como propósito:

1. descargar y cargar un dataset real de películas
2. inspeccionar su estructura y calidad
3. limpiar valores faltantes y duplicados
4. codificar variables categóricas
5. entrenar modelos de regresión lineal regularizada
6. entrenar un modelo de regresión logística para clasificación binaria
7. evaluar resultados con métricas apropiadas
8. guardar gráficos del proceso y del desempeño de los modelos

## Flujo de trabajo

### 1. Descarga y carga del dataset

Se usa `kagglehub` para descargar el dataset y detectar automáticamente el archivo CSV disponible.

### 2. Exploración gráfica y limpieza

El script genera:

- histogramas de variables numéricas
- matriz de correlación

Además realiza:

- eliminación de duplicados
- eliminación de filas sin valores en `RATING` o `RunTime`
- imputación de faltantes numéricos con la mediana
- imputación de faltantes categóricos con `Unknown`

### 3. Codificación de variables categóricas

Todas las columnas de tipo texto se convierten a formato numérico usando `LabelEncoder`.

### 4. Regresión lineal regularizada

Se selecciona `RATING` como variable objetivo para regresión.

Se entrenan dos modelos:

- `Ridge`
- `Lasso`

Ambos usan un `Pipeline` con:

- `StandardScaler`
- modelo de regresión correspondiente

Los hiperparámetros se ajustan mediante `RandomizedSearchCV` con validación cruzada de 5 particiones.

### 5. Regresión logística

Se crea una variable binaria llamada `rating_good`:

- `1`: película con `RATING >= 7.0`
- `0`: película con `RATING < 7.0`

Luego se entrena un modelo de `LogisticRegression` con:

- escalado de variables
- búsqueda aleatoria del parámetro `C`
- validación cruzada con métrica `F1`

## Métricas utilizadas

### Para regresión

- `R²`
- `MAE` (Mean Absolute Error)

### Para clasificación

- `Accuracy`
- `F1-Score`
- matriz de confusión

## Requisitos

Instala las dependencias con:

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn scipy
```

## Ejecución

Desde la raíz del proyecto:

```bash
python Lect5/lect5.py
```

O desde la carpeta [Lect5](Lect5):

```bash
python lect5.py
```

## Archivos generados

Todos los resultados se guardan en la carpeta [outputs](outputs).

### Visualizaciones exportadas

- `01_distribucion_variables.png`
- `02_matriz_correlacion.png`
- `03_train_test_split.png`
- `04_regresion_lineal_predicciones.png`
- `05_matriz_confusion.png`
- `06_regresion_logistica_predicciones.png`

## Estructura general del análisis

El script está organizado en estas etapas:

- **Carga del dataset**
- **Exploración gráfica y limpieza**
- **Codificación de variables**
- **Regresión lineal con Ridge y Lasso**
- **Regresión logística**
- **Resumen final**

## Modelos implementados

### `Ridge`

Modelo de regresión lineal con regularización $L_2$, útil cuando hay multicolinealidad o muchas variables correlacionadas.

### `Lasso`

Modelo de regresión lineal con regularización $L_1$, útil para reducir la complejidad del modelo y favorecer coeficientes más pequeños o nulos.

### `LogisticRegression`

Modelo de clasificación binaria utilizado para determinar si una película puede considerarse “buena” según su `RATING`.

## Conceptos cubiertos

Este ejercicio permite practicar:

- limpieza de datos
- imputación de valores faltantes
- codificación de variables categóricas
- separación de entrenamiento y prueba
- escalado de variables
- regularización en regresión
- búsqueda aleatoria de hiperparámetros
- validación cruzada
- evaluación de modelos supervisados
- análisis visual de predicciones

## Observaciones importantes

- El script reutiliza todas las columnas restantes como predictores, por lo que la calidad del modelo depende de la codificación aplicada.
- La variable `rating_good` se construye directamente desde `RATING`, por lo que si `RATING` permanece dentro de las variables predictoras, puede introducir fuga de información en la clasificación.
- La salida se guarda en la carpeta global [outputs](outputs), no dentro de [Lect5](Lect5).
- El archivo usa `RandomizedSearchCV`, así que el tiempo de ejecución puede variar según el equipo.

## Posibles mejoras

Algunas extensiones recomendadas para este trabajo:

1. excluir `RATING` de las variables predictoras en la regresión logística para evitar leakage
2. separar mejor las variables objetivo de clasificación y regresión
3. agregar curvas ROC y AUC
4. incluir métricas adicionales como `precision` y `recall`
5. comparar con modelos no lineales como `RandomForestRegressor` o `RandomForestClassifier`
6. guardar los mejores modelos entrenados en archivos `.joblib`

## Autor

Miguel Angel Mercado