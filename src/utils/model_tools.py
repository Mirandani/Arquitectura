# src/utils/model_tools.py
"""Modulo para generar métricas de evaluación de modelos de machine learning."""

import logging
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# Función para evaluar el modelo utilizando RMSE
def evaluar_modelo_rmse(modelo, x_validacion, y_validacion):
    """Evalúa el modelo utilizando RMSE.
    Args:
        modelo: El modelo a evaluar.
        x_validacion: Las características de validación.
        y_validacion: Las etiquetas de validación.
    Returns:
        El valor de RMSE.
    """
    try:
        prediccion = modelo.predict(x_validacion)
        error = np.sqrt(mean_squared_error(y_validacion, prediccion))
        return error
    except Exception as e:
        logger.error("Error al evaluar el modelo: %s", e)
        raise


def guardar_modelo(modelo, ruta):
    """Guarda el modelo utilizando joblib.
    Args:
        modelo: El modelo a guardar.
        ruta (str): La ruta del archivo donde se guardará el modelo.
    Returns:
        None
    """
    try:
        logger.info("Guardando el modelo en '%s'...", ruta)
        joblib.dump(modelo, ruta)
        logger.info("Modelo guardado exitosamente en '%s'.", ruta)
    except Exception as e:
        logger.error("Error al guardar el modelo: %s", e)
        raise


def cargar_modelo(ruta):
    """Carga un modelo guardado utilizando joblib.
    Args:
        ruta (str): La ruta del archivo del modelo a cargar.
    Returns:
        El modelo cargado.
    """
    try:
        logger.info("Cargando el modelo desde '%s'...", ruta)
        modelo = joblib.load(ruta)
        logger.info("Modelo cargado exitosamente desde '%s'.", ruta)
        return modelo
    except Exception as e:
        logger.error("Error al cargar el modelo: %s", e)
        raise


def guardar_predicciones(df, columnas, ruta):
    """Guarda predicciones en CSV e imprime confirmación.
    Args:
        df: DataFrame que contiene las predicciones.
        columnas: Lista de columnas a guardar.
        ruta: Ruta del archivo CSV donde se guardarán las predicciones.
    Returns:
        None
    """
    try:
        logger.info("Guardando predicciones en '%s'...", ruta)
        df[columnas].to_csv(ruta, index=False)
        logger.info("Predicciones guardadas exitosamente en '%s'.", ruta)
    except Exception as e:
        logger.error("Error al guardar las predicciones: %s", e)
        raise


def resumen_predicciones(predicciones):
    """Genera estadísticas descriptivas de las predicciones."""
    try:
        logger.info("Generando resumen de predicciones...")
        resumen = {
            "media": np.mean(predicciones),
            "mediana": np.median(predicciones),
            "min": np.min(predicciones),
            "max": np.max(predicciones),
        }
        logger.info("Resumen de predicciones: %s", resumen)
        return resumen
    except Exception as e:
        logger.error("Error al generar el resumen de predicciones: %s", e)
        raise
