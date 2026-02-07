# src/inference.py
"""Módulo para realizar inferencia utilizando un modelo de ML entrenado.

La entrada de este script son:
    - data/inference/datos_inferencia.parquet.
    - modelo entrenado en: modelo_random_forest.joblib.
La salida de este script son predicciones en batch que se guardan en:
    - data/predictions/predicciones_batch.csv
"""

import logging
import pandas as pd
from utils.model_tools import (
    cargar_modelo,
    guardar_predicciones,
    resumen_predicciones,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constantes
RUTA_MODELO_ENTRENADO = "artifacts/models/modelo_random_forest.joblib"
RUTA_DATOS_INFERENCIA = "data/inference/datos_inferencia.parquet"
RUTA_PREDICCIONES = "data/predictions/predicciones_batch.csv"

# Cargar datos de inferencia
datos_inferencia = pd.read_parquet(RUTA_DATOS_INFERENCIA)
datos_inferencia = datos_inferencia.drop(["item_cnt_month"], axis=1)

# Cargar el modelo entrenado
modelo_entrenado = cargar_modelo(RUTA_MODELO_ENTRENADO)

# Realizar predicciones batch
logger.info("Realizando predicciones...")
predicciones = modelo_entrenado.predict(datos_inferencia)
# Agregar predicciones al DataFrame
datos_inferencia["item_cnt_month_pred"] = predicciones

guardar_predicciones(datos_inferencia, ["ID", "item_cnt_month_pred"], RUTA_PREDICCIONES)

# metricas de resumen
resumen = resumen_predicciones(predicciones)
logger.info(
    "Predicciones - media: %.2f, mediana: %.2f, min: %.2f, max: %.2f",
    resumen["media"],
    resumen["mediana"],
    resumen["min"],
    resumen["max"],
)
