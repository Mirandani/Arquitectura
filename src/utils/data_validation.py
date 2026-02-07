# src/utils/data_validation.py
"""Módulo para validación de datos."""

import logging

logger = logging.getLogger(__name__)


# función para validación de datos nulos y vista previa
def validar_datos(df, nombre="DataFrame"):
    """Reporte de dimensiones, nulos y vista previa.
    Args:
        df (pd.DataFrame): El DataFrame a validar.
        nombre (str): Nombre descriptivo del DataFrame para el reporte.
    Returns:
        None
    """
    try:
        logger.info("\n--- Validando %s ---", nombre)
        logger.info("Dimensiones: %s", df.shape)
        nulos = df.isnull().sum()
        nulos = nulos[nulos > 0]
        logger.info("Nulos:\n%s", nulos if not nulos.empty else "Ninguno")
        logger.info("\nHead (3):")
        logger.info("\n%s", df.head(3))
    except Exception as e:
        logger.error("Error al validar %s: %s", nombre, e)
        raise
