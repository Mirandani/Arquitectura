# src/utils/model_tools.py
"""Modulo para generar métricas de evaluación de modelos de machine learning.
"""
import numpy as np
from sklearn.metrics import mean_squared_error
import logging
import joblib
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Función para evaluar el modelo utilizando RMSE
def evaluar_modelo_rmse(modelo, X_validacion, Y_validacion):
    """Evalúa el modelo utilizando RMSE.
    Args:
        modelo: El modelo a evaluar.
        X_validacion: Las características de validación.
        Y_validacion: Las etiquetas de validación.
    Returns:
        El valor de RMSE.
    """
    try:
        prediccion = modelo.predict(X_validacion)
        error = np.sqrt(mean_squared_error(Y_validacion, prediccion))
        return error
    except Exception as e:
        logger.error(f"Error al evaluar el modelo: {e}")
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
        logger.info(f"Guardando el modelo en '{ruta}'...")
        joblib.dump(modelo, ruta)
        logger.info(f"Modelo guardado exitosamente en '{ruta}'.")
    except Exception as e:
        logger.error(f"Error al guardar el modelo: {e}")
        raise


def cargar_modelo(ruta):
    """Carga un modelo guardado utilizando joblib.
    Args:
        ruta (str): La ruta del archivo del modelo a cargar.
    Returns:
        El modelo cargado.
    """
    try:
        logger.info(f"Cargando el modelo desde '{ruta}'...")
        modelo = joblib.load(ruta)
        logger.info(f"Modelo cargado exitosamente desde '{ruta}'.")
        return modelo
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        raise
