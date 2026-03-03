# src/inference/__main__.py
"""Script principal para realizar inferencia utilizando un modelo de ML entrenado."""

import argparse
from inference.inference import (
    main,
    RUTA_MODELO_ENTRENADO,
    RUTA_DATOS_INFERENCIA,
    RUTA_PREDICCIONES,
)

parser = argparse.ArgumentParser(description="Inferencia batch")
parser.add_argument("--modelo", default=RUTA_MODELO_ENTRENADO)
parser.add_argument("--datos", default=RUTA_DATOS_INFERENCIA)
parser.add_argument("--salida", default=RUTA_PREDICCIONES)
args = parser.parse_args()

main(args.modelo, args.datos, args.salida)
