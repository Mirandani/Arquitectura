"""Tests unitarios para el módulo de preprocesamiento"""

import pytest
import pandas as pd
from preprocessing.prep import generar_grid_base, agregar_historia


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def df_ventas_grid():
    """Dataset simulado para probar la generación de la matriz base"""
    return pd.DataFrame({
        "date_block_num": [0, 0, 0],
        "shop_id": [1, 2, 1],   # 2 tiendas únicas
        "item_id": [10, 10, 20] # 2 productos únicos
    })

@pytest.fixture
def df_ventas_historia():
    """Dataset simulado para probar la creación de variables de meses anteriores"""
    return pd.DataFrame({
        "date_block_num": [0, 1, 2],
        "shop_id": [1, 1, 1],
        "item_id": [10, 10, 10],
        "ventas": [100, 200, 300]
    })


# ── Tests: generar_grid_base ──────────────────────────────────────────────────

def test_generar_grid_base_columnas_correctas(df_ventas_grid):
    resultado = generar_grid_base(df_ventas_grid)
    assert list(resultado.columns) == ["date_block_num", "shop_id", "item_id"]

def test_generar_grid_base_crea_producto_cartesiano(df_ventas_grid):
    resultado = generar_grid_base(df_ventas_grid)
    assert len(resultado) == 4
    assert resultado["shop_id"].nunique() == 2
    assert resultado["item_id"].nunique() == 2


# ── Tests: agregar_historia ───────────────────────────────────────────────────

def test_agregar_historia_crea_nueva_columna(df_ventas_historia):
    resultado = agregar_historia(df_ventas_historia, meses_atras=[1], columna_base="ventas")
    assert "ventas_mes_ant_1" in resultado.columns

def test_agregar_historia_desplaza_valores(df_ventas_historia):
    resultado = agregar_historia(df_ventas_historia, meses_atras=[1], columna_base="ventas")
    # Extraemos el valor del mes anterior para el registro del mes 1
    venta_pasada = resultado.loc[resultado["date_block_num"] == 1, "ventas_mes_ant_1"].values[0]
    assert venta_pasada == 100.0

def test_agregar_historia_multiples_meses(df_ventas_historia):
    resultado = agregar_historia(df_ventas_historia, meses_atras=[1, 2], columna_base="ventas")
    assert "ventas_mes_ant_1" in resultado.columns
    assert "ventas_mes_ant_2" in resultado.columns 
    venta_pasada_2 = resultado.loc[resultado["date_block_num"] == 2, "ventas_mes_ant_2"].values[0]
    assert venta_pasada_2 == 100.0