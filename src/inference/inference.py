# src/inference/inference.py
"""Módulo para realizar inferencia utilizando un modelo de ML entrenado.

La entrada de este script son:
    - data/inference/datos_inferencia.parquet.
    - modelo entrenado en: modelo_random_forest.joblib.
La salida de este script son predicciones en batch que se guardan en:
    - data/predictions/predicciones_batch.csv
"""

import pandas as pd
from utils.model_tools import (
    cargar_modelo,
    guardar_predicciones as _guardar_predicciones,
    resumen_predicciones as _resumen_predicciones,
)
from utils.logger import configurar_logger

# Constantes
RUTA_MODELO_ENTRENADO = "artifacts/models/modelo_random_forest.joblib"
RUTA_DATOS_INFERENCIA = "data/inference/datos_inferencia.parquet"
RUTA_PREDICCIONES = "data/predictions/predicciones_batch.csv"

def caragar_datos_inferencia(ruta_datos: str) -> pd.DataFrame:
    """Carga los datos de inferencia desde un archivo Parquet."""
    logger = configurar_logger(__name__)
    try:
        logger.info("Cargando datos de inferencia desde: %s", ruta_datos)
        datos = pd.read_parquet(ruta_datos)
        logger.info("Datos cargados exitosamente: %s registros", format(len(datos), ","))
        return datos
    except Exception as e:
        logger.error("Error al cargar los datos de inferencia: %s", e)
        raise

def preparar_datos_inferencia(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara los datos de inferencia eliminando la columna objetivo si está presente."""
    logger = configurar_logger(__name__)
    if "item_cnt_month" in df.columns:
        df = df.drop(["item_cnt_month"], axis=1)
        logger.info("Columna 'item_cnt_month' eliminada de los datos de inferencia")
    return df

def cargar_modelo_entrenado(ruta_modelo: str):
    """Carga el modelo entrenado desde la ruta especificada."""
    logger = configurar_logger(__name__)
    logger.info("Cargando modelo entrenado desde: %s", ruta_modelo)
    modelo = cargar_modelo(ruta_modelo)
    logger.info("Modelo cargado exitosamente")
    return modelo

def generar_predicciones(modelo, datos):
    """Genera predicciones utilizando el modelo cargado."""
    logger = configurar_logger(__name__)
    logger.info("Realizando predicciones batch...")
    predicciones = modelo.predict(datos)
    logger.info("Predicciones completadas: %s valores", len(predicciones))
    return predicciones

def guardar_predicciones(datos, columnas, ruta_salida):
    """Guarda las predicciones en un archivo CSV."""
    logger = configurar_logger(__name__)
    logger.info("Guardando predicciones en: %s", ruta_salida)
    _guardar_predicciones(datos, columnas, ruta_salida)
    logger.info("Predicciones guardadas exitosamente en: %s", ruta_salida)

def resumen_predicciones(predicciones):
    """Genera un resumen estadístico de las predicciones."""
    logger = configurar_logger(__name__)
    logger.info("Generando resumen estadístico de las predicciones...")
    resumen = _resumen_predicciones(predicciones)
    logger.info(
        "Resumen predicciones - Media: %.2f | Mediana: %.2f | Min: %.2f | Max: %.2f",
        resumen["media"],
        resumen["mediana"],
        resumen["min"],
        resumen["max"],
    )
    return resumen

def main(
    ruta_modelo: str = RUTA_MODELO_ENTRENADO,
    ruta_datos:  str = RUTA_DATOS_INFERENCIA,
    ruta_salida: str = RUTA_PREDICCIONES,
) -> None:

    # Configuración de logging
    logger = configurar_logger(__name__)
    logger.info("=== Inicio del proceso de inferencia ===")

    # Cargar datos de inferencia
    datos_inferencia = caragar_datos_inferencia(ruta_datos)

    # Preparar datos de inferencia (eliminar columna objetivo si existe)
    datos_inferencia = preparar_datos_inferencia(datos_inferencia)

    # Cargar el modelo entrenado
    modelo_entrenado = cargar_modelo_entrenado(ruta_modelo)

    # Realizar predicciones batch
    logger.info("Realizando predicciones batch...")
    predicciones = generar_predicciones(modelo_entrenado, datos_inferencia)

    # Agregar predicciones al DataFrame de inferencia
    datos_inferencia["item_cnt_month_pred"] = predicciones
    logger.info("Predicciones completadas: %s valores", len(predicciones))

    # Guardar predicciones en CSV
    guardar_predicciones(
        datos=datos_inferencia,
        columnas=["ID", "item_cnt_month_pred"],
        ruta_salida=ruta_salida,
    )

    # Métricas de resumen
    resumen = resumen_predicciones(predicciones)
    logger.info(
        "Resumen predicciones - Media: %.2f | Mediana: %.2f | Min: %.2f | Max: %.2f",
        resumen["media"],
        resumen["mediana"],
        resumen["min"],
        resumen["max"],
    )

    logger.info("=== Proceso de inferencia finalizado exitosamente ===")
