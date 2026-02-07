# src/utils/dtypes.py
"""Módulo para optimización de tipos de datos en DataFrames."""

import numpy as np


# función para cambiar tipos de datos
def optimizar_tipos(df):
    """Optimiza los tipos de datos del DataFrame para reducir el uso de memoria.
    Args:
        df (pd.DataFrame): El DataFrame a optimizar.
    Returns:
        pd.DataFrame: El DataFrame con tipos de datos optimizados.
    """
    return df.assign(
        date_block_num=lambda x: x["date_block_num"].astype(np.int8),
        shop_id=lambda x: x["shop_id"].astype(np.int8),
        item_id=lambda x: x["item_id"].astype(np.int16),
    )
