# src/train.py
"""Módulo para entrenamiento de modelo de machine learning.

La entrada del script son:
    - data/prep/datos_entreno.parquet
    - data/prep/datos_validacion.parquet
La salida del script es un modelo entrenado:
    - artifacts/models/modelo_random_forest.joblib
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from utils.model_tools import evaluar_modelo_rmse, guardar_modelo
from utils.logger import configurar_logger

# Constantes
PATH_DATOS_ENTRENAMIENTO = "data/prep/datos_entreno.parquet"
PATH_DATOS_VALIDACION = "data/prep/datos_validacion.parquet"
PATH_MODELO_ENTRENADO = "artifacts/models/modelo_random_forest.joblib"


if __name__ == "__main__":
    # Configuración de logging
    logger = configurar_logger(__name__)
    logger.info("=== Inicio del proceso de entrenamiento de modelo ===")

    # Carga de datos de entrenamiento y validación
    logger.info("Cargando datasets de entrenamiento y validación...")
    datos_entrenamiento = pd.read_parquet(PATH_DATOS_ENTRENAMIENTO)
    datos_validacion = pd.read_parquet(PATH_DATOS_VALIDACION)
    logger.info(
        "Datos cargados - Entrenamiento: %s registros, Validación: %s registros",
        format(len(datos_entrenamiento), ","),
        format(len(datos_validacion), ","),
    )

    # Preparación de datos para entrenamiento
    logger.info("Preparando features y target...")
    X_entreno = datos_entrenamiento.drop(["item_cnt_month"], axis=1)
    Y_entreno = datos_entrenamiento["item_cnt_month"]

    # Preparación de datos para validación
    X_validacion = datos_validacion.drop(["item_cnt_month"], axis=1)
    Y_validacion = datos_validacion["item_cnt_month"]
    logger.info("Features: %s columnas", X_entreno.shape[1])

    # Entrenamiento del modelo de regresion lineal
    logger.info("Iniciando entrenamiento del modelo de Regresión Lineal...")
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X_entreno, Y_entreno)
    error_lineal = evaluar_modelo_rmse(modelo_lineal, X_validacion, Y_validacion)
    logger.info("✓ RMSE Regresión Lineal: %.4f", error_lineal)

    # Entrenamiento del modelo Random Forest
    logger.info("Iniciando entrenamiento del modelo Random Forest...")
    modelo_random_forest = RandomForestRegressor(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
    )
    modelo_random_forest.fit(X_entreno, Y_entreno)
    error_random_forest = evaluar_modelo_rmse(
        modelo_random_forest, X_validacion, Y_validacion
    )
    logger.info("✓ RMSE Random Forest: %.4f", error_random_forest)

    # Guardado del modelo entrenado
    logger.info("Guardando modelo entrenado...")
    guardar_modelo(modelo_random_forest, PATH_MODELO_ENTRENADO)
    logger.info("Modelo guardado en: %s", PATH_MODELO_ENTRENADO)

    logger.info("=== Proceso de entrenamiento finalizado exitosamente ===")
