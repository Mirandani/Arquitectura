# src/inference/test/test_inference.py
"""Test para el módulo de inferencia."""

import pytest
import numpy as np
import pandas as pd
from inference.inference import (
    preparar_datos_inferencia,
    cargar_modelo_entrenado,
    generar_predicciones,
    guardar_predicciones,
    resumen_predicciones,
)


class ModeloMock:
    """Modelo dummy que predice siempre 1.0 — no necesita .joblib"""

    def predict(self, X):
        return np.ones(len(X))


@pytest.fixture
def df_con_target():
    """Dataset simulado que incluye la columna target 'item_cnt_month' """
    return pd.DataFrame(
        {
            "feature_1": [10, 20, 30],
            "feature_2": [0.1, 0.2, 0.3],
            "item_cnt_month": [5, 10, 15],  # columna target
        }
    )


def test_preparar_datos_elimina_columna_target():
    """Prueba que la función eliminar la columna target 'item_cnt_month' si existe"""
    # Creo el DataFrame en memoria, sin tocar ningún archivo
    df = pd.DataFrame(
        {
            "feature_1": [10, 20],
            "item_cnt_month": [5, 10],  # esta debe desaparecer
        }
    )
    resultado = preparar_datos_inferencia(df)
    assert "item_cnt_month" not in resultado.columns


def test_preparar_datos_sin_columna_target():
    """Prueba que si no hay columna target, el DataFrame se devuelva igual"""
    df = pd.DataFrame(
        {
            "feature_1": [10, 20],
            "feature_2": [0.1, 0.2],
        }
    )
    resultado = preparar_datos_inferencia(df)
    assert resultado.equals(df)  # no debe modificar el DataFrame


def test_generar_predicciones_con_modelo_mock():
    """Prueba que el modelo mock genere predicciones de 1.0 para cualquier input"""
    modelo = ModeloMock()
    df = pd.DataFrame(
        {
            "feature_1": [10, 20],
            "feature_2": [0.1, 0.2],
        }
    )
    predicciones = generar_predicciones(modelo, df)
    assert np.array_equal(predicciones, np.ones(len(df)))


def test_resumen_predicciones():
    """Prueba que el resumen de predicciones calcule correctamente media, mediana, min y max"""
    predicciones = np.array([1, 2, 3, 4, 5])
    resumen = resumen_predicciones(predicciones)
    assert resumen["media"] == 3.0
    assert resumen["mediana"] == 3.0
    assert resumen["min"] == 1
    assert resumen["max"] == 5


def test_guardar_predicciones(tmp_path):
    df = pd.DataFrame(
        {
            "ID": [1, 2],
            "item_cnt_month_pred": [0.5, 1.5],
        }
    )
    ruta_salida = tmp_path / "predicciones_test.csv"
    guardar_predicciones(df, ["ID", "item_cnt_month_pred"], ruta_salida)

    # Verificar que el archivo se creó y tiene el contenido correcto
    assert ruta_salida.exists()
    df_leido = pd.read_csv(ruta_salida)
    assert df_leido.equals(df[["ID", "item_cnt_month_pred"]])
