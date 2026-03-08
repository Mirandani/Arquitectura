"""_summary_
La entrada del script son datos en la ruta especificada por CLI.
La salida del script son datos procesados en la ruta especificada por CLI.
Este modulo sirve para hacer las transformaciones necesarias para dejar
los datos listos para el analisis exploratorio y modelado.
"""

import time
from itertools import product
import pandas as pd
import numpy as np

# Importación de módulos de utils
from utils.outputs import guardar_dataset
from utils.dtypes import optimizar_tipos
from utils.data_validation import validar_datos
from utils.logger import configurar_logger

pd.set_option("display.float_format", lambda x: f"{x:.2f}")

# Constantes de rutas de archivos

PATH_ITEMS = "data/raw/items_en.csv"
PATH_CATEGORIES = "data/raw/item_categories_en.csv"
PATH_SHOPS = "data/raw/shops_en.csv"
PATH_TRAIN = "data/raw/sales_train.csv"
PATH_TEST = "data/raw/test.csv"

PATH_OUT_TRAIN = "data/prep/datos_entreno.parquet"
PATH_OUT_VAL = "data/prep/datos_validacion.parquet"
PATH_OUT_INFER = "data/inference/datos_inferencia.parquet"

# FUNCIONES ESPECÍFICAS

# función para generar la matriz base de mes-tienda-item


def generar_grid_base(df_entrenamiento):
    """Genera las combinaciones de mes-tienda-item."""
    grid = []
    grid_cols = ["date_block_num", "shop_id", "item_id"]

    # Usamos el máximo mes en los datos + iteramos
    meses = df_entrenamiento["date_block_num"].unique()

    for i in meses:
        ventas = df_entrenamiento[df_entrenamiento.date_block_num == i]
        grid.append(
            np.array(
                list(
                    product(
                        [i],
                        ventas.shop_id.unique(),
                        ventas.item_id.unique(),
                    )
                ),
                dtype="int16",
            )
        )

    return pd.DataFrame(np.vstack(grid), columns=grid_cols)


# función para agregar historia de ventas
def agregar_historia(datos, meses_atras, columna_base):
    """Agrega variables de meses anteriores al dataframe."""
    df_temp = datos[["date_block_num", "shop_id", "item_id", columna_base]]

    for mes in meses_atras:
        desplazado = df_temp.copy()
        desplazado.columns = [
            "date_block_num",
            "shop_id",
            "item_id",
            f"{columna_base}_mes_ant_{mes}",
        ]
        desplazado["date_block_num"] += mes

        datos = pd.merge(
            datos, desplazado, on=["date_block_num", "shop_id", "item_id"], how="left"
        )
    return datos


###########################################################################
# EJECUCIÓN PRINCIPAL
###########################################################################


def main(
    items_path: str,
    categories_path: str,
    shops_path: str,
    train_path: str,
    test_path: str,
    out_train: str,
    out_val: str,
    out_infer: str,
):
    """
    Ejecuta el pipeline principal de preprocesamiento de datos.
    """

    # Configuración inicial de logger
    logger = configurar_logger(__name__)
    start_time = time.time()
    logger.info("Iniciando proceso de preparación de datos...")

    try:
        # Lectura de datos
        logger.info("Cargando datasets raw...")
        # traducción de nombres de artículos a inglés
        articulos = pd.read_csv(items_path)
        # traducción de categorías a inglés
        categorias = pd.read_csv(categories_path)
        tiendas = pd.read_csv(shops_path)

        # Join de dataframes
        datos_entrenamiento = (
            pd.read_csv(train_path)
            .merge(articulos, on="item_id", how="left")
            .merge(categorias, on="item_category_id", how="left")
            .merge(tiendas, on="shop_id", how="left")
            .assign(
                date=lambda df: pd.to_datetime(df["date"], format="%d.%m.%Y"),
                month=lambda df: df["date"].dt.month,
            )
        )
        logger.info(
            "Datos cargados exitosamente: %s registros",
            format(len(datos_entrenamiento), ","),
        )
        # Validación inicial de datos
        validar_datos(datos_entrenamiento, "Datos Raw")

        # ##########################################################################
        # LIMPIEZA DE DATOS
        # ##########################################################################

        logger.info("Iniciando limpieza de datos...")
        filas_antes = len(datos_entrenamiento)

        datos_entrenamiento = (
            datos_entrenamiento.query("item_price > 0")  # Eliminar precios negativos
            .query("item_price < 100000")  # Eliminar precios muy altos
            .query("item_cnt_day < 1000")  # Eliminar ventas diarias excesivas
            .drop_duplicates()
        )

        # Warning si se eliminaron filas
        filas_despues = len(datos_entrenamiento)
        filas_eliminadas = filas_antes - filas_despues

        if filas_eliminadas > 0:
            logger.warning(
                "Se eliminaron %s registros durante la limpieza",
                format(filas_eliminadas, ","),
            )

        logger.info(
            "Dimensiones después de limpieza: %s",
            datos_entrenamiento.shape,
        )

        # ##########################################################################
        # CONSOLIDACIÓN DE INFORMACIÓN MES-> TIENDA -> PRODUCTO -> VENTAS
        # ##########################################################################

        logger.info("Generando grid base, combinaciones mes-tienda-item...")

        # matriz_ventas matriz con mes-tienda-item
        matriz_ventas = generar_grid_base(datos_entrenamiento)

        matriz_ventas = matriz_ventas.pipe(optimizar_tipos).sort_values(
            ["date_block_num", "shop_id", "item_id"]
        )

        # Incluimos las ventas por mes
        ventas_agrupadas = (
            datos_entrenamiento.groupby(["date_block_num", "shop_id", "item_id"])
            .agg(item_cnt_month=("item_cnt_day", "sum"))
            .reset_index()
        )

        # Unimos ventas con matriz
        # Reemplazamos nulos por 0
        # Limitamos a 20 como pide la competencia

        cols = ["date_block_num", "shop_id", "item_id"]

        matriz_ventas = pd.merge(
            matriz_ventas, ventas_agrupadas, on=cols, how="left"
        ).assign(
            item_cnt_month=lambda df: (
                df["item_cnt_month"].fillna(0).clip(0, 20).astype(np.float32)
            )
        )

        logger.info(
            "Matriz consolidada generada. Dimensiones: %s",
            matriz_ventas.shape,
        )

        # ##########################################################################
        #   MATRIZ MES TIENDA PRODUCTO + DATOS_PRUEBA + HISTORIA
        # ##########################################################################

        # Preparar datos_prueba para unión con matriz

        logger.info("Integrando datos de prueba...")

        datos_prueba = (
            pd.read_csv(test_path).assign(date_block_num=34).pipe(optimizar_tipos)
        )

        # Unión de datos_prueba con matriz consolidada
        matriz_ventas = pd.concat(
            [matriz_ventas, datos_prueba], ignore_index=True, sort=False
        ).fillna(0)

        # variables con número de mes

        logger.info("Generando variables de meses de historia ...")

        matriz_ventas = matriz_ventas.pipe(
            agregar_historia, meses_atras=[1, 2, 3, 12], columna_base="item_cnt_month"
        ).fillna(0)

        cols_historia = [f"item_cnt_month_mes_ant_{m}" for m in [1, 2, 3, 12]]
        for col in cols_historia:
            matriz_ventas[col] = matriz_ventas[col].astype(np.float32) 

        # Corrección base
        logger.info("Recuperando características y limpiando columnas...")

        if "ID" in matriz_ventas.columns:
            matriz_ventas = matriz_ventas.drop("ID", axis=1)

        # Recuperar la categoría del artículo
        matriz_ventas = pd.merge(
            matriz_ventas, 
            articulos[["item_id", "item_category_id"]], 
            on="item_id", 
            how="left"
        )
            
        # ### GUARDANDO DATASETS

        # División de datos
        logger.info("Guardando datasets procesados...")

        guardar_dataset(matriz_ventas[matriz_ventas.date_block_num < 33], out_train)
        guardar_dataset(matriz_ventas[matriz_ventas.date_block_num == 33], out_val)
        guardar_dataset(matriz_ventas[matriz_ventas.date_block_num == 34], out_infer)

        # Logger de tiempo de ejecución
        duration = time.time() - start_time
        logger.info(
            "Proceso finalizado con éxito. Tiempo de ejecución: %.2f segundos",
            duration,
        )

    # Logger de errores críticos
    except Exception as e:
        logger.error(
            "Fallo crítico en el script de preparación: %s",
            str(e),
            exc_info=True,
        )
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preparación de datos de ventas para modelo"
    )

    # Entradas
    parser.add_argument(
        "--items", type=str, default=PATH_ITEMS, help="Ruta al archivo items_en.csv"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=PATH_CATEGORIES,
        help="Ruta al archivo categorias",
    )
    parser.add_argument(
        "--shops", type=str, default=PATH_SHOPS, help="Ruta al archivo tiendas"
    )
    parser.add_argument(
        "--train", type=str, default=PATH_TRAIN, help="Ruta al archivo de entrenamiento"
    )
    parser.add_argument(
        "--test", type=str, default=PATH_TEST, help="Ruta al archivo test"
    )

    # Salidas
    parser.add_argument(
        "--out-train",
        type=str,
        default=PATH_OUT_TRAIN,
        help="Ruta para datos de entrenamiento",
    )
    parser.add_argument(
        "--out-val",
        type=str,
        default=PATH_OUT_VAL,
        help="Ruta para datos de validacion",
    )
    parser.add_argument(
        "--out-infer",
        type=str,
        default=PATH_OUT_INFER,
        help="Ruta para datos de inferencia",
    )

    args = parser.parse_args()

    main(
        items_path=args.items,
        categories_path=args.categories,
        shops_path=args.shops,
        train_path=args.train,
        test_path=args.test,
        out_train=args.out_train,
        out_val=args.out_val,
        out_infer=args.out_infer,
    )

    