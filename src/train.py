# src/train.py
""" Módulo para entrenamiento de modelo de machine learning.

    La entrada del script son:
        - data/prep/datos_entreno.parquet
        - data/prep/datos_validacion.parquet
    La salida del script es un modelo entrenado:
        - artifacts/models/modelo_random_forest.joblib
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.utils.model_tools import evaluar_modelo_rmse, guardar_modelo

# Configuración de logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constantes
PATH_DATOS_ENTRENAMIENTO = 'data/prep/datos_entreno.parquet'
PATH_DATOS_VALIDACION = 'data/prep/datos_validacion.parquet'
PATH_MODELO_ENTRENADO = 'artifacts/models/modelo_random_forest.joblib'

#  Carga de datos de entrenamiento y validación
datos_entrenamiento = pd.read_parquet(PATH_DATOS_ENTRENAMIENTO)
datos_validacion = pd.read_parquet(PATH_DATOS_VALIDACION)
# datos_prueba_final = pd.read_parquet('data/prep/datos_prueba_final.parquet')

# Preparación de datos para entrenamiento
X_entreno = datos_entrenamiento.drop(['item_cnt_month'], axis=1)
Y_entreno = datos_entrenamiento['item_cnt_month']

# Preparación de datos para validación
X_validacion = datos_validacion.drop(['item_cnt_month'], axis=1)
Y_validacion = datos_validacion['item_cnt_month']

# Entrenamiento del modelo de regresion lineal
modelo_lineal = LinearRegression()
logger.info("Iniciando el entrenamiento del modelo de regresión lineal...")
modelo_lineal.fit(X_entreno, Y_entreno)

# Evaluación del modelo de regresión lineal
error_lineal = (
    evaluar_modelo_rmse(
        modelo_lineal,
        X_validacion,
        Y_validacion)
    )
logger.info(f"RMSE Regresión Lineal: {error_lineal:.4f}")

# Entrenamiento del modelo Random Forest
modelo_random_forest = (
    RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1)
    )
logger.info("Iniciando el entrenamiento del modelo Random Forest...")
modelo_random_forest.fit(X_entreno, Y_entreno)

# Evaluación del modelo Random Forest
error_random_forest = (
    evaluar_modelo_rmse(
        modelo_random_forest,
        X_validacion,
        Y_validacion)
    )
logger.info(f"RMSE Random Forest: {error_random_forest:.4f}")


# Guardado del modelo entrenado
guardar_modelo(modelo_random_forest, PATH_MODELO_ENTRENADO)


def main():
    logger.info("Inicio del proceso de entrenamiento de modelo.")
    # Aquí se podría agregar más lógica si es necesario
    logger.info("Proceso de entrenamiento de modelo finalizado.")


if __name__ == "__main__":
    main()
