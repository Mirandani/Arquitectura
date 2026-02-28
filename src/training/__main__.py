# src/training/__main__.py
"""Punto de entrada CLI para el módulo de entrenamiento."""

import argparse
from training.train import (
    main,
    MODELOS_DISPONIBLES,
    PATH_DATOS_ENTRENAMIENTO,
    PATH_DATOS_VALIDACION,
    PATH_MODELO_ENTRENADO,
)

parser = argparse.ArgumentParser(description="Entrenamiento de modelo de ML")

parser.add_argument("--entrada",      default=PATH_DATOS_ENTRENAMIENTO,
                    help="Ruta al parquet de entrenamiento")
parser.add_argument("--validacion",   default=PATH_DATOS_VALIDACION,
                    help="Ruta al parquet de validación")
parser.add_argument("--salida",       default=PATH_MODELO_ENTRENADO,
                    help="Ruta donde guardar el modelo (.joblib)")

parser.add_argument("--modelo-baseline", default="linear",
                    choices=MODELOS_DISPONIBLES,
                    help="Tipo de modelo baseline (para comparación)")
parser.add_argument("--modelo-principal", default="random_forest",
                    choices=MODELOS_DISPONIBLES,
                    help="Tipo de modelo principal (a guardar)")

parser.add_argument("--n-estimators", default=50,  type=int,
                    help="Número de árboles (solo random_forest)")
parser.add_argument("--max-depth",    default=10,  type=int,
                    help="Profundidad máxima (solo random_forest)")
parser.add_argument("--random-state", default=42,  type=int,
                    help="Semilla para reproducibilidad")

args = parser.parse_args()

main(
    entrada          = args.entrada,
    validacion       = args.validacion,
    salida           = args.salida,
    modelo_baseline  = args.modelo_baseline,
    modelo_principal = args.modelo_principal,
    n_estimators     = args.n_estimators,
    max_depth        = args.max_depth,
    random_state     = args.random_state,
)
