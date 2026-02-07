# src/inference.py
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
    guardar_predicciones,
    resumen_predicciones,
)
from utils.logger import configurar_logger

# Constantes
RUTA_MODELO_ENTRENADO = "artifacts/models/modelo_random_forest.joblib"
RUTA_DATOS_INFERENCIA = "data/inference/datos_inferencia.parquet"
RUTA_PREDICCIONES = "data/predictions/predicciones_batch.csv"


if __name__ == "__main__":
    # Configuración de logging
    logger = configurar_logger(__name__)
    logger.info("=== Inicio del proceso de inferencia ===")

    # Cargar datos de inferencia
    logger.info("Cargando datos de inferencia...")
    datos_inferencia = pd.read_parquet(RUTA_DATOS_INFERENCIA)
    logger.info("Datos cargados: %s registros", format(len(datos_inferencia), ","))
    datos_inferencia = datos_inferencia.drop(["item_cnt_month"], axis=1)

    # Cargar el modelo entrenado
    logger.info("Cargando modelo entrenado desde: %s", RUTA_MODELO_ENTRENADO)
    modelo_entrenado = cargar_modelo(RUTA_MODELO_ENTRENADO)
    logger.info("Modelo cargado exitosamente")

    # Realizar predicciones batch
    logger.info("Realizando predicciones batch...")
    predicciones = modelo_entrenado.predict(datos_inferencia)
    datos_inferencia["item_cnt_month_pred"] = predicciones
    logger.info("Predicciones completadas: %s valores", len(predicciones))

    # Guardar predicciones
    logger.info("Guardando predicciones en: %s", RUTA_PREDICCIONES)
    guardar_predicciones(
        datos_inferencia,
        ["ID", "item_cnt_month_pred"],
        RUTA_PREDICCIONES,
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
