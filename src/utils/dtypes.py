import pandas as pd
import numpy as np

# función para cambiar tipos de datos
def optimizar_tipos(df):
    return df.assign(
        date_block_num=lambda x: x['date_block_num'].astype(np.int8),
        shop_id=lambda x: x['shop_id'].astype(np.int8),
        item_id=lambda x: x['item_id'].astype(np.int16)
    )