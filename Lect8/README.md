# Clasificación de tensión financiera con Gradient Boosting y Random Forest

## Descripción

Este proyecto implementa un flujo de análisis exploratorio, preprocesamiento y clasificación supervisada sobre el dataset [Lect8/dataset_sintetico_FIRE_UdeA_realista.csv](Lect8/dataset_sintetico_FIRE_UdeA_realista.csv). El script principal [Lect8/lect8.py](Lect8/lect8.py) trabaja con una variable objetivo binaria llamada `label`, que representa la presencia o ausencia de tensión financiera.

El análisis combina:

- exploración visual del dataset
- separación temporal de entrenamiento y prueba
- preprocesamiento para variables numéricas y categóricas
- ajuste de hiperparámetros con `GridSearchCV`
- comparación entre `GradientBoostingClassifier` y `RandomForestClassifier`
- interpretación del problema con importancia de variables, árbol de decisión y PCA

## Objetivo

El propósito del script es construir un pipeline de clasificación para identificar casos de tensión financiera usando información tabular del contexto FIRE UdeA.

En particular, el flujo busca:

1. cargar y revisar el dataset
2. separar entrenamiento y prueba por año
3. explorar distribuciones y relaciones entre variables
4. entrenar modelos de clasificación binaria
5. comparar desempeño con métricas y curvas diagnósticas
6. generar visualizaciones interpretables para el análisis

## Dataset

**Archivo principal:** [Lect8/dataset_sintetico_FIRE_UdeA_realista.csv](Lect8/dataset_sintetico_FIRE_UdeA_realista.csv)

### Variables clave

- `label`: variable objetivo binaria
- `anio`: variable temporal usada para dividir entrenamiento y prueba
- `unidad`: variable categórica
- variables financieras numéricas como `liquidez`, `dias_efectivo`, `cfo`, entre otras

## Metodología general

## 1. Carga de datos

El script carga el dataset con `pandas`:

- usa `label` como variable objetivo
- define como características todas las columnas excepto `label`
- convierte `anio` a entero para asegurar la división temporal correcta

## 2. División entrenamiento/prueba por tiempo

No se usa una partición aleatoria estándar. En cambio, la separación se hace por año:

- **entrenamiento:** observaciones con `anio <= 2022`
- **prueba:** observaciones con `anio > 2022`

Esto simula un escenario más realista de predicción hacia periodos futuros.

## 3. Análisis exploratorio de datos

El script genera varias visualizaciones para comprender el comportamiento del dataset.

### Exploración básica

- `info()` del dataframe
- estadísticas descriptivas
- conteo de valores faltantes

### Visualizaciones EDA

- distribución de la variable objetivo
- histogramas para variables seleccionadas
- violin plots por clase
- box plots por clase
- strip plots por clase
- swarm plots por clase
- matriz de correlación de variables numéricas

## 4. Preprocesamiento

Se separan las variables en:

- **categóricas:** `unidad`
- **numéricas:** todas las demás columnas predictoras

### Pipeline numérico

- imputación de faltantes con mediana
- escalado con `StandardScaler`

### Pipeline categórico

- imputación con valor más frecuente
- codificación con `OneHotEncoder(handle_unknown="ignore")`

Ambos pipelines se integran mediante `ColumnTransformer`.

## 5. Modelos entrenados

### `GradientBoostingClassifier`

Se optimiza con `GridSearchCV` usando:

- `n_estimators`
- `learning_rate`
- `max_depth`
- `subsample`

La búsqueda usa:

- validación cruzada de 3 folds
- métrica de selección `f1`

### `RandomForestClassifier`

Se usa como modelo base fuerte con parámetros fijos:

- `n_estimators=500`
- `min_samples_leaf=2`
- `random_state=42`
- `n_jobs=-1`

## 6. Métricas de evaluación

Los modelos se evalúan sobre el conjunto de prueba con:

- `Accuracy`
- `F1-score`
- `ROC-AUC`
- matriz de confusión

Además se generan curvas complementarias:

- curva ROC comparativa
- curva Precision-Recall comparativa
- calibration plot

## 7. Interpretabilidad

El script añade varias salidas para interpretar el problema:

### Importancia de variables

A partir del `RandomForestClassifier`, se calcula la importancia Gini de las variables y se grafica el top 20.

### Árbol de decisión interpretable

Se entrena un `DecisionTreeClassifier` poco profundo (`max_depth=3`) usando solo variables numéricas imputadas, con el fin de visualizar reglas simples de decisión.

### PCA 2D

Se proyecta el espacio transformado de características a 2 componentes principales para observar la separación entre clases.

## Dependencias

Instala las librerías necesarias con:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Ejecución

Desde la raíz del proyecto:

```bash
python Lect8/lect8.py
```

O desde la carpeta [Lect8](Lect8):

```bash
python lect8.py
```

## Archivos generados

El script guarda las salidas gráficas en la misma carpeta [Lect8](Lect8).

### Archivos EDA

- [Lect8/eda_label_balance.png](Lect8/eda_label_balance.png)
- [Lect8/eda_hist_liquidez.png](Lect8/eda_hist_liquidez.png)
- [Lect8/eda_hist_dias_efectivo.png](Lect8/eda_hist_dias_efectivo.png)
- [Lect8/eda_hist_cfo.png](Lect8/eda_hist_cfo.png)
- [Lect8/eda_violin_liquidez.png](Lect8/eda_violin_liquidez.png)
- [Lect8/eda_violin_dias_efectivo.png](Lect8/eda_violin_dias_efectivo.png)
- [Lect8/eda_violin_cfo.png](Lect8/eda_violin_cfo.png)
- [Lect8/eda_box_liquidez.png](Lect8/eda_box_liquidez.png)
- [Lect8/eda_box_dias_efectivo.png](Lect8/eda_box_dias_efectivo.png)
- [Lect8/eda_box_cfo.png](Lect8/eda_box_cfo.png)
- [Lect8/eda_strip_liquidez.png](Lect8/eda_strip_liquidez.png)
- [Lect8/eda_strip_dias_efectivo.png](Lect8/eda_strip_dias_efectivo.png)
- [Lect8/eda_strip_cfo.png](Lect8/eda_strip_cfo.png)
- [Lect8/eda_swarm_liquidez.png](Lect8/eda_swarm_liquidez.png)
- [Lect8/eda_swarm_dias_efectivo.png](Lect8/eda_swarm_dias_efectivo.png)
- [Lect8/eda_swarm_cfo.png](Lect8/eda_swarm_cfo.png)
- [Lect8/eda_correlacion_numericas.png](Lect8/eda_correlacion_numericas.png)

### Archivos de evaluación y modelado

- [Lect8/cm_gradient_boosting.png](Lect8/cm_gradient_boosting.png)
- [Lect8/cm_random_forest.png](Lect8/cm_random_forest.png)
- [Lect8/rf_importancia_variables.png](Lect8/rf_importancia_variables.png)
- [Lect8/arbol_decision_FIRE_UdeA.png](Lect8/arbol_decision_FIRE_UdeA.png)
- [Lect8/pca_fire_udea.png](Lect8/pca_fire_udea.png)
- [Lect8/roc_comparativa.png](Lect8/roc_comparativa.png)
- [Lect8/pr_comparativa.png](Lect8/pr_comparativa.png)
- [Lect8/calibration_plot.png](Lect8/calibration_plot.png)

## Conceptos cubiertos

Este ejercicio permite practicar:

- clasificación supervisada binaria
- partición temporal de datos
- análisis exploratorio de datos
- imputación de faltantes
- escalado y codificación categórica
- `ColumnTransformer` y `Pipeline`
- optimización de hiperparámetros con `GridSearchCV`
- comparación de modelos ensemble
- matrices de confusión
- curvas ROC y Precision-Recall
- calibración de probabilidades
- reducción de dimensionalidad con PCA
- interpretabilidad con árboles e importancia de variables

## Observaciones importantes

- La división entrenamiento/prueba se basa en tiempo, no en aleatoriedad.
- La variable `anio` se usa para partir los datos y también permanece dentro de las features salvo que el usuario la excluya manualmente.
- El análisis EDA detallado se concentra en `liquidez`, `dias_efectivo` y `cfo` cuando esas columnas existen.
- Las imágenes se guardan en la carpeta local [Lect8](Lect8), no en [outputs](outputs).
- `GradientBoostingClassifier` puede tardar más debido a la búsqueda en malla.

## Posibles mejoras

1. guardar métricas en un archivo CSV o TXT
2. exportar el mejor modelo entrenado con `joblib`
3. añadir validación temporal más robusta
4. evaluar balance de clases y usar técnicas de reponderación
5. incorporar SHAP para explicabilidad local y global
6. automatizar la generación de un reporte final
7. comparar con `XGBoost` o `LightGBM`

## Autor

Miguel Angel Mercado