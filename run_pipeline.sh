#!/bin/bash
# ============================================================================
# SCRIPT DE EJECUCIÓN - PIPELINE COMPLETO LECT 2, 3 Y 4
# ============================================================================
# 
# Este script facilita la ejecución del pipeline completo que integra:
#   • Lección 2: Machine Learning + Feature Engineering
#   • Lección 3: Análisis Avanzado de Datos
#   • Lección 4: Exploración Gráfica y Transformaciones
#
# Uso:
#   sh run_pipeline.sh
#
# ============================================================================

echo "================================================================================"
echo "EJECUTANDO PIPELINE COMPLETO - LECCIONES 2, 3 Y 4"
echo "================================================================================"
echo ""

# Rutas
FDAA_DIR="/Users/miguelmercado/Documents/7mo Semestre/FDAA"
LECT4_DIR="$FDAA_DIR/Lect4"
VENV_PYTHON="$FDAA_DIR/path/to/venv/bin/python"

# Verificar que Python virtual environment existe
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ ERROR: No se encontró el entorno virtual en: $VENV_PYTHON"
    exit 1
fi

echo "✓ Entorno virtual configurado"
echo "✓ Ejecutando pipeline..."
echo ""

# Ejecutar el pipeline
cd "$LECT4_DIR" || exit
"$VENV_PYTHON" movies_complete_pipeline.py

# Verificar si la ejecución fue exitosa
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ PIPELINE EJECUTADO EXITOSAMENTE"
    echo ""
    echo "Archivos generados en: $FDAA_DIR/outputs/"
    echo ""
    echo "Visualizaciones:"
    echo "  • pipeline_histogramas.png"
    echo "  • pipeline_matriz_correlación.png"
    echo "  • pipeline_comparación_modelos.png"
    echo "  • pipeline_predicciones_vs_reales.png"
    echo "  • pipeline_pca.png"
    echo ""
    echo "Reportes:"
    echo "  • pipeline_reporte_completo.txt"
    echo "  • pipeline_resultados_modelos.csv"
    echo ""
    echo "Modelos:"
    echo "  • best_model_movies.joblib"
    echo ""
else
    echo ""
    echo "❌ ERROR: El pipeline falló"
    exit 1
fi

echo "================================================================================"
