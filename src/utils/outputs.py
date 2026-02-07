"""Módulo para operaciones de salida de datos."""

import logging

logger = logging.getLogger(__name__)


def guardar_dataset(df, ruta):
    """Guarda el dataframe en parquet e imprime confirmación.

    Args:
        df: DataFrame de pandas a guardar.
        ruta (str): Ruta del archivo parquet donde se guardará el dataset.

    Returns:
        None
    """
    try:
        df.to_parquet(ruta, index=False)
        logger.info("Guardado: %s  Dimensiones: %s", ruta, df.shape)

    except Exception as e:
        logger.error("Error al guardar %s: %s", ruta, e)
        raise
