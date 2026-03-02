# src/training/test/test_train.py
"""Tests unitarios para el módulo de entrenamiento."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from training.train import preparar_datos_entrenamiento_validacion, construir_modelo


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def df_entreno():
    return pd.DataFrame({
        "feature_1":      [1, 2, 3, 4, 5],
        "feature_2":      [0.1, 0.2, 0.3, 0.4, 0.5],
        "item_cnt_month": [10, 20, 30, 40, 50],
    })

@pytest.fixture
def df_validacion():
    return pd.DataFrame({
        "feature_1":      [6, 7],
        "feature_2":      [0.6, 0.7],
        "item_cnt_month": [60, 70],
    })


# ── Tests: preparar_datos_entrenamiento_validacion ────────────────────────────

def test_preparar_datos_elimina_target_de_features(df_entreno, df_validacion):
    X_e, _, X_v, _ = preparar_datos_entrenamiento_validacion(df_entreno, df_validacion)
    assert "item_cnt_month" not in X_e.columns
    assert "item_cnt_month" not in X_v.columns

def test_preparar_datos_conserva_features(df_entreno, df_validacion):
    X_e, _, X_v, _ = preparar_datos_entrenamiento_validacion(df_entreno, df_validacion)
    assert "feature_1" in X_e.columns
    assert "feature_2" in X_v.columns

def test_preparar_datos_target_correcto(df_entreno, df_validacion):
    _, Y_e, _, Y_v = preparar_datos_entrenamiento_validacion(df_entreno, df_validacion)
    assert list(Y_e) == [10, 20, 30, 40, 50]
    assert list(Y_v) == [60, 70]

def test_preparar_datos_lanza_error_sin_target():
    df_sin_target = pd.DataFrame({"feature_1": [1, 2], "feature_2": [3, 4]})
    with pytest.raises(ValueError, match="item_cnt_month"):
        preparar_datos_entrenamiento_validacion(df_sin_target, df_sin_target)


# ── Tests: construir_modelo ───────────────────────────────────────────────────

def test_construir_modelo_random_forest_retorna_tipo_correcto():
    modelo = construir_modelo("random_forest")
    assert isinstance(modelo, RandomForestRegressor)

def test_construir_modelo_linear_retorna_tipo_correcto():
    modelo = construir_modelo("linear")
    assert isinstance(modelo, LinearRegression)

def test_construir_modelo_random_forest_respeta_hiperparametros():
    modelo = construir_modelo("random_forest", n_estimators=200, max_depth=6, random_state=7)
    assert modelo.n_estimators == 200
    assert modelo.max_depth    == 6
    assert modelo.random_state == 7

def test_construir_modelo_tipo_invalido_lanza_error():
    with pytest.raises(ValueError, match="no soportado"):
        construir_modelo("xgboost")
