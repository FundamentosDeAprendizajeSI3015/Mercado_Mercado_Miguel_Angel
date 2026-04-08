# Análisis de clases latentes y casos sospechosos en FIRE UdeA

## Descripción

Este proyecto documenta el script [Lect10/generar_respuestas_u dea.py](Lect10/generar_respuestas_u%20dea.py), cuyo objetivo es analizar un dataset tabular del contexto FIRE UdeA para identificar estructuras latentes, contrastarlas con la etiqueta original `label` y detectar observaciones potencialmente sospechosas.

El flujo combina técnicas de:

- preprocesamiento de datos
- reducción de dimensionalidad
- modelado probabilístico con mezclas gaussianas
- análisis de densidad con `DBSCAN`
- visualización exploratoria en 2D y 3D
- generación de métricas resumidas en formato JSON

## Objetivo

El propósito principal del script es:

1. cargar el dataset y conservar solo variables numéricas
2. descubrir dos clases latentes con `GaussianMixture`
3. reinterpretar esas clases como estados “saludables” o “de mayor riesgo” a partir de variables financieras clave
4. comparar la clase latente con la etiqueta original `label`
5. marcar registros inconsistentes o sospechosos
6. estudiar si esos sospechosos caen en núcleos, bordes u outliers de `DBSCAN`
7. generar visualizaciones útiles para responder preguntas analíticas sobre el dataset

## Dataset utilizado

El script trabaja con:

- [Lect10/dataset_sintetico_FIRE_UdeA%20Lect9%20copy.csv](Lect10/dataset_sintetico_FIRE_UdeA%20Lect9%20copy.csv)

Además genera resultados en:

- [Lect10/metricas_respuestas_u_dea.json](Lect10/metricas_respuestas_u_dea.json)
- [Lect10/visualizaciones_u_dea](Lect10/visualizaciones_u_dea)

## Flujo general del análisis

## 1. Carga de datos

El script lee el CSV desde la carpeta [Lect10](Lect10) y separa:

- `y_label`: la etiqueta original del dataset
- `Xdf`: variables predictoras numéricas, excluyendo `label`

Solo se conservan columnas numéricas para simplificar el modelado de mezclas gaussianas y la reducción de dimensionalidad.

## 2. Preprocesamiento

Antes del análisis, se aplican dos pasos:

- imputación de faltantes con `SimpleImputer(strategy="mean")`
- estandarización con `StandardScaler`

Esto genera una matriz `Xz` con variables escaladas, adecuada para modelos basados en distancia y covarianza.

## 3. Descubrimiento de clases latentes con `GaussianMixture`

Se entrena un modelo:

- `GaussianMixture(n_components=2, covariance_type="full", random_state=42)`

El modelo asigna a cada observación una clase latente inicial (`latent`).

Posteriormente, el script calcula los promedios por cluster en tres variables financieras:

- `liquidez`
- `cfo`
- `dias_efectivo`

Con esas medias se construye un `health_score` para identificar cuál cluster representa una condición financiera más saludable.

A partir de esto se redefinen las clases como:

- **Clase 1:** cluster más saludable
- **Clase 2:** cluster menos saludable

## 4. Comparación con la etiqueta original

El script evalúa cómo se distribuyen las observaciones cuya etiqueta original cumple `label == 1` dentro de las clases latentes encontradas.

Se calculan métricas como:

- total de registros con `label == 1`
- cuántos de ellos caen en Clase 1
- cuántos caen en Clase 2
- porcentajes correspondientes

## 5. Detección de casos sospechosos

Se define un registro sospechoso cuando existe discordancia entre la etiqueta original y la clase latente inferida:

- `label == 1` pero clase latente desfavorable
- `label == 0` pero clase latente saludable

Esto genera una máscara booleana `suspected` para estudiar casos potencialmente inconsistentes, ambiguos o interesantes para revisión.

## 6. Análisis de densidad con `DBSCAN`

Luego se ajusta un modelo:

- `DBSCAN(eps=3.1410, min_samples=5)`

El objetivo no es redefinir las clases, sino ubicar los casos sospechosos dentro de la estructura local de densidad:

- **core points**
- **border points**
- **outliers** (`label = -1` en DBSCAN)

Se reporta cuántos sospechosos caen en cada categoría y sus porcentajes.

## 7. Visualizaciones generadas

El script genera varias gráficas para interpretar la estructura del dataset.

### PCA 2D

Se proyectan los datos en dos componentes principales y se colorean por:

- clase latente
- etiqueta original

Archivo:
- [Lect10/visualizaciones_u_dea/01_pca2d_clase_latente.png](Lect10/visualizaciones_u_dea/01_pca2d_clase_latente.png)

### PCA 3D

Se construye una visualización tridimensional del espacio reducido, coloreada por clase latente.

Archivo:
- [Lect10/visualizaciones_u_dea/02_pca3d_clases.png](Lect10/visualizaciones_u_dea/02_pca3d_clases.png)

### t-SNE 2D

Se proyectan los datos con `TSNE` y se resaltan los puntos sospechosos.

Archivo:
- [Lect10/visualizaciones_u_dea/03_tsne_sospechosos.png](Lect10/visualizaciones_u_dea/03_tsne_sospechosos.png)

### Pairplot de variables clave

Si existen suficientes variables relevantes, se genera un pairplot con variables como:

- `liquidez`
- `dias_efectivo`
- `cfo`
- `hhi_fuentes`
- `gastos_personal`

Archivo:
- [Lect10/visualizaciones_u_dea/04_pairplot_variables_clave.png](Lect10/visualizaciones_u_dea/04_pairplot_variables_clave.png)

## Métricas generadas

El script exporta un resumen a [Lect10/metricas_respuestas_u_dea.json](Lect10/metricas_respuestas_u_dea.json).

Entre los campos reportados se incluyen:

- `total_l1`
- `class1_in_l1`
- `class2_in_l1`
- `p_class1_in_l1`
- `p_class2_in_l1`
- `suspected_n`
- `suspected_core`
- `suspected_border`
- `suspected_outliers`
- `suspected_core_pct`
- `suspected_border_pct`
- `suspected_outliers_pct`
- `pca2_explained_pct`
- `visual_dir`

## Dependencias

Instala las librerías necesarias con:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Ejecución

Desde la raíz del proyecto:

```bash
python "Lect10/generar_respuestas_u dea.py"
```

O desde la carpeta [Lect10](Lect10):

```bash
python "generar_respuestas_u dea.py"
```

## Archivos de salida

### Resultado principal

- [Lect10/metricas_respuestas_u_dea.json](Lect10/metricas_respuestas_u_dea.json)

### Visualizaciones

- [Lect10/visualizaciones_u_dea/01_pca2d_clase_latente.png](Lect10/visualizaciones_u_dea/01_pca2d_clase_latente.png)
- [Lect10/visualizaciones_u_dea/02_pca3d_clases.png](Lect10/visualizaciones_u_dea/02_pca3d_clases.png)
- [Lect10/visualizaciones_u_dea/03_tsne_sospechosos.png](Lect10/visualizaciones_u_dea/03_tsne_sospechosos.png)
- [Lect10/visualizaciones_u_dea/04_pairplot_variables_clave.png](Lect10/visualizaciones_u_dea/04_pairplot_variables_clave.png)

## Conceptos cubiertos

Este ejercicio permite practicar:

- clustering probabilístico con `GaussianMixture`
- estandarización e imputación de datos
- detección de inconsistencias entre etiquetas observadas y estructura latente
- clustering basado en densidad con `DBSCAN`
- análisis de puntos core, borde y outliers
- reducción de dimensionalidad con PCA
- visualización no lineal con t-SNE
- análisis visual de variables financieras

## Observaciones importantes

- El script depende de que existan variables como `liquidez`, `cfo` y `dias_efectivo` para construir el criterio de salud financiera.
- Solo se usan variables numéricas; cualquier información categórica se descarta.
- `GaussianMixture` encuentra componentes latentes, no clases supervisadas verdaderas.
- Los casos sospechosos no necesariamente son errores; pueden ser ejemplos frontera, ruido o cambios de patrón.
- La elección de `eps = 3.1410` en `DBSCAN` está fijada manualmente y puede requerir ajuste si cambia el dataset.

## Posibles mejoras

1. parametrizar el número de componentes del modelo GMM
2. comparar con más de dos clases latentes
3. usar probabilidades posteriores del GMM en lugar de etiquetas duras
4. automatizar la selección de `eps` para `DBSCAN`
5. guardar un reporte textual adicional con interpretación de resultados
6. incluir columnas categóricas usando un pipeline mixto
7. añadir validaciones para comprobar que existen las variables financieras clave

## Autor

Miguel Angel Mercado