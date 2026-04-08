# Análisis estadístico y transformación de un dataset de películas

## Descripción

Este proyecto ejecuta un análisis exploratorio y de preprocesamiento sobre un dataset de películas descargado desde Kaggle. El script [Lect4/lect4.py](Lect4/lect4.py) calcula métricas descriptivas, detecta y elimina outliers, genera visualizaciones, transforma variables categóricas y aplica diferentes técnicas de escalado y transformación logarítmica.

El objetivo es dejar el dataset listo para análisis posteriores o para su uso en modelos de machine learning.

## Dataset utilizado

El script descarga automáticamente el dataset:

- **Fuente:** Kaggle
- **Identificador:** `bharatnatrayn/movies-dataset-for-feature-extracion-prediction`
- **Archivo principal:** `movies.csv`

## Funcionalidades principales

El script realiza las siguientes tareas:

1. **Descarga y carga del dataset** usando `kagglehub`.
2. **Análisis descriptivo inicial**:
   - dimensiones del dataset
   - primeras filas
   - tipos de datos
   - estadísticas básicas
3. **Medidas de tendencia central**:
   - media
   - mediana
   - moda
4. **Medidas de dispersión**:
   - desviación estándar
   - varianza
   - rango
   - coeficiente de variación
5. **Medidas de posición y detección de outliers** con el método IQR.
6. **Eliminación de outliers** en todas las columnas numéricas.
7. **Generación de histogramas** con asimetría y curtosis.
8. **Gráficos de dispersión** entre variables numéricas con línea de tendencia y correlación.
9. **Transformación de variables categóricas**:
   - One Hot Encoding
   - Label Encoding
10. **Matriz de correlación** y detección de variables altamente correlacionadas.
11. **Escalado de datos**:
   - `MinMaxScaler`
   - `StandardScaler`
12. **Transformación logarítmica** para columnas numéricas positivas.
13. **Generación automática de conclusiones** en un archivo de texto.

## Requisitos

### Dependencias de Python

Instala las librerías necesarias con:

```bash
pip install kagglehub pandas numpy matplotlib seaborn scipy scikit-learn
```

### Requisitos adicionales

Como el dataset se descarga desde Kaggle, debes tener acceso configurado para la descarga. Si tu entorno requiere autenticación de Kaggle, asegúrate de tener tus credenciales configuradas correctamente.

## Cómo ejecutar

Desde la raíz del proyecto:

```bash
python Lect4/lect4.py
```

O desde la carpeta `Lect4`:

```bash
python lect4.py
```

## Salidas generadas

Todos los resultados se guardan en la carpeta [outputs](outputs), ubicada en la raíz del workspace.

### Archivos esperados

- `histogramas_distribuciones.png`
- `graficos_dispersión.png`
- `matriz_correlación.png`
- `transformacion_log_*.png` para cada columna transformada
- `analisis_conclusiones.txt`

## Estructura del flujo del script

El archivo [Lect4/lect4.py](Lect4/lect4.py) está dividido en secciones:

- **Sección 1:** descarga y carga del dataset
- **Sección 2:** medidas de tendencia central
- **Sección 3:** medidas de dispersión
- **Sección 4:** medidas de posición y outliers
- **Sección 5:** histogramas
- **Sección 6:** gráficos de dispersión
- **Sección 7:** transformaciones de columnas
- **Sección 8:** correlación
- **Sección 9:** escalado
- **Sección 10:** transformación logarítmica
- **Sección 11:** conclusiones finales

## Conceptos aplicados

Este ejercicio cubre varios temas de análisis de datos y preprocesamiento:

- estadística descriptiva
- detección de valores atípicos
- visualización de distribuciones
- análisis de relaciones entre variables
- codificación de variables categóricas
- correlación lineal
- normalización y estandarización
- transformación de variables sesgadas

## Posibles usos académicos

Este script puede servir como base para:

- talleres de análisis exploratorio de datos
- ejercicios de limpieza y preparación de datasets
- prácticas de estadística aplicada
- preparación de datos para modelos supervisados o no supervisados

## Observaciones

- El script crea automáticamente la carpeta de salida si no existe.
- Los archivos se guardan en la carpeta global [outputs](outputs), no dentro de [Lect4](Lect4).
- La eliminación de outliers se aplica de forma secuencial por cada columna numérica, por lo que el número final de filas puede reducirse considerablemente.
- La transformación logarítmica solo se aplica a columnas con valores estrictamente positivos.

## Autor

Miguel Angel Mercado