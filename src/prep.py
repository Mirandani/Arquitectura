"""_summary_
La entrada del script son datos data/raw.
La salida del script son datos data/prep.
Este modulo sirve para hacer las transformaciones necesarias para dejar
los datos listos para el analisis exploratorio y modelado.
"""

import os
import time
import logging
from datetime import datetime
from itertools import product
import pandas as pd
import numpy as np


# Importación de módulos de utils

from utils.outputs import guardar_dataset
from utils.dtypes import optimizar_tipos
from utils.data_validation import validar_datos


pd.set_option('display.float_format', lambda x: f'{x:.2f}')

# Configuración de logger
def configurar_logger():
    """Configura el logger para guardar en archivo y mostrar en consola"""
    log_dir = 'artifacts/logs'
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'{log_dir}/prep_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# FUNCIONES ESPECÍFICAS

# función para generar la matriz base de mes-tienda-item

def generar_grid_base(df_entrenamiento):
    """Genera las combinaciones de mes-tienda-item."""
    grid = []
    grid_cols = ['date_block_num', 'shop_id', 'item_id']

    # Usamos el máximo mes en los datos + iteramos
    meses = df_entrenamiento['date_block_num'].unique()

    for i in meses:
        ventas = df_entrenamiento[df_entrenamiento.date_block_num == i]
        grid.append(np.array(list(product(
            [i], ventas.shop_id.unique(), ventas.item_id.unique()
        )), dtype='int16'))

    return pd.DataFrame(np.vstack(grid), columns=grid_cols)

# función para agregar historia de ventas
def agregar_historia(datos, meses_atras, columna_base):
    """Agrega variables de meses anteriores al dataframe."""
    df_temp = datos[['date_block_num', 'shop_id', 'item_id', columna_base]]

    for mes in meses_atras:
        desplazado = df_temp.copy()
        desplazado.columns = [
            'date_block_num', 'shop_id', 'item_id', f"{columna_base}_mes_ant_{mes}"
        ]
        desplazado['date_block_num'] += mes

        datos = pd.merge(
            datos, desplazado, on=['date_block_num', 'shop_id', 'item_id'], how='left'
        )
    return datos



###########################################################################
# EJECUCIÓN PRINCIPAL
###########################################################################

if __name__ == "__main__":
    # Configuración inicial de logger
    logger = configurar_logger()
    start_time = time.time()
    logger.info("Iniciando proceso de preparación de datos...")

    try:
        # Lectura de datos
        logger.info("Cargando datasets raw...")
        # traducción de nombres de artículos a inglés
        articulos = pd.read_csv('data/raw/items_en.csv')
        # traducción de categorías a inglés
        categorias = pd.read_csv('data/raw/item_categories_en.csv')
        tiendas = pd.read_csv('data/raw/shops_en.csv')

        # Join de dataframes
        datos_entrenamiento = (
            pd.read_csv('data/raw/sales_train.csv')
            .merge(articulos, on='item_id', how='left')
            .merge(categorias, on='item_category_id', how='left')
            .merge(tiendas, on='shop_id', how='left')
            .assign(
                date=lambda df: pd.to_datetime(df['date'], format='%d.%m.%Y'),
                month=lambda df: df['date'].dt.month
            )
        )
        logger.info(
            "Datos cargados exitosamente: %s registros",
            format(len(datos_entrenamiento), ",")
        )
        # Validación inicial de datos
        validar_datos(datos_entrenamiento, "Datos Raw")



        ###########################################################################
        # LIMPIEZA DE DATOS
        ###########################################################################

        logger.info("Iniciando limpieza de datos...")
        filas_antes = len(datos_entrenamiento)

        datos_entrenamiento = (
            datos_entrenamiento
            .query('item_price > 0')           # Eliminar precios negativos
            .query('item_price < 100000')      # Eliminar precios muy altos
            .query('item_cnt_day < 1000')      # Eliminar ventas diarias excesivas
            .drop_duplicates()
        )

        # Warning si se eliminaron filas
        filas_despues = len(datos_entrenamiento)
        filas_eliminadas = filas_antes - filas_despues

        if filas_eliminadas > 0:
            logger.warning(
                "Se eliminaron %s registros durante la limpieza",
                format(filas_eliminadas, ",")
            )

        logger.info("Dimensiones después de limpieza: %s", datos_entrenamiento.shape)


        ###########################################################################
        # CONSOLIDACIÓN DE INFORMACIÓN MES-> TIENDA -> PRODUCTO -> VENTAS
        ###########################################################################

        logger.info("Generando grid base, combinaciones mes-tienda-item...")

        # matriz_ventas matriz con mes-tienda-item
        matriz_ventas = generar_grid_base(datos_entrenamiento)

        matriz_ventas = (
            matriz_ventas
            .pipe(optimizar_tipos)
            .sort_values(['date_block_num', 'shop_id', 'item_id'])
        )

        # Incluimos las ventas por mes
        ventas_agrupadas = (
            datos_entrenamiento
            .groupby(['date_block_num', 'shop_id', 'item_id'])
            .agg(item_cnt_month=('item_cnt_day', 'sum'))
            .reset_index()
        )

        # Unimos ventas con matriz
        # Reemplazamos nulos por 0
        # Limitamos a 20 como pide la competencia

        cols = ['date_block_num', 'shop_id', 'item_id']

        matriz_ventas = (
            pd.merge(matriz_ventas, ventas_agrupadas, on=cols, how='left')
            .assign(
                item_cnt_month=lambda df: df['item_cnt_month']
                .fillna(0).clip(0, 20).astype(np.float16)
            )
        )

        logger.info(
            "Matriz consolidada generada. Dimensiones: %s",
            matriz_ventas.shape
        )


        ###########################################################################
        #   MATRIZ MES TIENDA PRODUCTO + DATOS_PRUEBA + HISTORIA
        ###########################################################################

        # Preparar datos_prueba para unión con matriz

        logger.info("Integrando datos de prueba...")

        datos_prueba = (
            pd.read_csv('data/raw/test.csv')
            .assign(date_block_num=34)
            .pipe(optimizar_tipos)
        )

        # Unión de datos_prueba con matriz consolidada
        matriz_ventas = pd.concat(
            [matriz_ventas, datos_prueba], ignore_index=True, sort=False
        ).fillna(0)


        # variables con número de mes

        logger.info("Generando variables de meses de historia ...")

        matriz_ventas = (
            matriz_ventas
            .pipe(agregar_historia, meses_atras=[1, 2, 3, 12], columna_base='item_cnt_month')
            .fillna(0)
        )

        ##### GUARDANDO DATASETS

        # División de datos
        logger.info("Guardando datasets procesados...")

        guardar_dataset(
            matriz_ventas[matriz_ventas.date_block_num < 33],
            'data/prep/datos_entreno.parquet'
        )

        guardar_dataset(
            matriz_ventas[matriz_ventas.date_block_num == 33],
            'data/prep/datos_validacion.parquet'
        )

        guardar_dataset(
            matriz_ventas[matriz_ventas.date_block_num == 34],
            'data/inference/datos_inferencia.parquet'
        )


        # Logger de tiempo de ejecución
        duration = time.time() - start_time
        logger.info(
            "Proceso finalizado con éxito. Tiempo de ejecución: %.2f segundos",
            duration
        )


    # Logger de errores críticos
    except Exception as e:
        logger.error(
            "Fallo crítico en el script de preparación: %s",
            str(e),
            exc_info=True
        )
        raise
