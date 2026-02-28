# src/training/train.py
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

# Constantes — valores por defecto; pueden sobreescribirse mediante argumentos CLI
PATH_DATOS_ENTRENAMIENTO = "data/prep/datos_entreno.parquet"
PATH_DATOS_VALIDACION    = "data/prep/datos_validacion.parquet"
PATH_MODELO_ENTRENADO    = "artifacts/models/modelo_random_forest.joblib"

MODELOS_DISPONIBLES = ["random_forest", "linear"]


def preparar_datos_entrenamiento_validacion(
    datos_entrenamiento: pd.DataFrame,
    datos_validacion: pd.DataFrame,
    columna_target: str = "item_cnt_month",
):
    """Separa features y target en los datasets de entrenamiento y validación.

    Args:
        datos_entrenamiento: DataFrame con datos de entrenamiento.
        datos_validacion:    DataFrame con datos de validación.
        columna_target:      Nombre de la columna objetivo.
    Returns:
        Tupla (X_entreno, Y_entreno, X_validacion, Y_validacion).
    Raises:
        ValueError: Si la columna target no está presente en alguno de los datasets.
    """
    logger = configurar_logger(__name__)

    for nombre, df in [("entrenamiento", datos_entrenamiento), ("validación", datos_validacion)]:
        if columna_target not in df.columns:
            raise ValueError(
                f"Columna target '{columna_target}' no encontrada en datos de {nombre}."
            )

    X_entreno    = datos_entrenamiento.drop([columna_target], axis=1)
    Y_entreno    = datos_entrenamiento[columna_target]
    X_validacion = datos_validacion.drop([columna_target], axis=1)
    Y_validacion = datos_validacion[columna_target]

    logger.info(
        "Features preparados — Entrenamiento: %s cols, Validación: %s cols",
        X_entreno.shape[1],
        X_validacion.shape[1],
    )
    return X_entreno, Y_entreno, X_validacion, Y_validacion


def construir_modelo(
    tipo_modelo: str,
    n_estimators: int = 50,
    max_depth: int    = 10,
    random_state: int = 42,
):
    """Construye y retorna el modelo según el tipo especificado.

    Args:
        tipo_modelo:   Tipo de modelo ('random_forest' o 'linear').
        n_estimators:  Número de árboles (solo para random_forest).
        max_depth:     Profundidad máxima (solo para random_forest).
        random_state:  Semilla para reproducibilidad (solo para random_forest).
    Returns:
        Instancia del modelo sklearn lista para entrenar.
    Raises:
        ValueError: Si el tipo de modelo no está soportado.
    """
    if tipo_modelo == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
    elif tipo_modelo == "linear":
        return LinearRegression()
    else:
        raise ValueError(
            f"Modelo '{tipo_modelo}' no soportado. Opciones: {MODELOS_DISPONIBLES}"
        )


def main(
    entrada:              str = PATH_DATOS_ENTRENAMIENTO,
    validacion:           str = PATH_DATOS_VALIDACION,
    salida:               str = PATH_MODELO_ENTRENADO,
    modelo_baseline:      str = "linear",
    modelo_principal:     str = "random_forest",
    n_estimators:         int = 50,
    max_depth:            int = 10,
    random_state:         int = 42,
) -> None:
    logger = configurar_logger(__name__)
    logger.info("=== Inicio del proceso de entrenamiento de modelo ===")

    # Carga de datos
    datos_entrenamiento = pd.read_parquet(entrada)
    datos_validacion    = pd.read_parquet(validacion)
    logger.info(
        "Datos cargados — Entrenamiento: %s registros, Validación: %s registros",
        format(len(datos_entrenamiento), ","),
        format(len(datos_validacion), ","),
    )

    # Separar features y target
    X_entreno, Y_entreno, X_validacion, Y_validacion = (
        preparar_datos_entrenamiento_validacion(datos_entrenamiento, datos_validacion)
    )
    logger.info("Features: %s columnas", X_entreno.shape[1])

    # Modelo baseline — para comparación de desempeño
    logger.info("Entrenando modelo baseline — %s...", modelo_baseline)
    model_base = construir_modelo(
        tipo_modelo=modelo_baseline,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model_base.fit(X_entreno, Y_entreno)
    error_base = evaluar_modelo_rmse(model_base, X_validacion, Y_validacion)
    logger.info("✓ RMSE %s (baseline): %.4f", modelo_baseline, error_base)

    # Modelo principal — el que se guarda
    logger.info("Entrenando modelo principal — %s...", modelo_principal)
    modelo = construir_modelo(
        tipo_modelo=modelo_principal,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    modelo.fit(X_entreno, Y_entreno)
    error = evaluar_modelo_rmse(modelo, X_validacion, Y_validacion)
    logger.info("✓ RMSE %s (principal): %.4f", modelo_principal, error)

    # Comparación
    mejora = ((error_base - error) / error_base) * 100
    logger.info(
        "Mejora respecto al baseline: %.2f%% %s",
        abs(mejora),
        "↓" if mejora > 0 else "↑",
    )

    # Guardar el modelo principal
    guardar_modelo(modelo, salida)
    logger.info("Modelo principal guardado en: %s", salida)

    logger.info("=== Proceso de entrenamiento finalizado exitosamente ===")
