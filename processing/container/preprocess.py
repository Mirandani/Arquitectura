# processing/container/preprocess.py

"""
Script de preprocesamiento para SageMaker Processing Job.
Lee datos crudos desde /opt/ml/processing/input/
Procesa y guarda en /opt/ml/processing/output/
Estructura de entrada esperada:
- items_en.csv
- item_categories_en.csv
- shops_en.csv
- sales_train.csv
- test.csv
Estructura de salida generada:
- train/datos_entreno.parquet
- validation/datos_validacion.parquet
- test/datos_inferencia.parquet
El script realiza:
1. Lectura de datos crudos
2. Limpieza de datos (eliminar outliers, duplicados)
3. Consolidación en matriz mes-tienda-producto-ventas
4. Integración de datos de prueba
5. Generación de variables de historia (ventas meses anteriores)
6. Guardado de datasets procesados en formato parquet
"""

import sys
import os
import argparse
import logging
import time
from itertools import product
import pandas as pd
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

pd.set_option("display.float_format", lambda x: f"{x:.2f}")

# Rutas por defecto de SageMaker
DEFAULT_INPUT_PATH = "/opt/ml/processing/input"
DEFAULT_OUTPUT_PATH = "/opt/ml/processing/output"


# ========================================================================
# FUNCIONES AUXILIARES
# ========================================================================

def optimizar_tipos(df):
    """Optimiza los tipos de datos del dataframe."""
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    return df


def guardar_dataset(df, path):
    """Guarda dataset en formato parquet."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Dataset guardado en: {path} (shape: {df.shape})")


def validar_datos(df, nombre="Dataset"):
    """Valida que no haya problemas básicos en los datos."""
    logger.info(f"Validando {nombre}: {df.shape}")
    logger.info(f"  - Columnas: {list(df.columns)}")
    logger.info(f"  - Nulls: {df.isnull().sum().sum()}")


def generar_grid_base(df_entrenamiento):
    """Genera las combinaciones de mes-tienda-item."""
    grid = []
    grid_cols = ["date_block_num", "shop_id", "item_id"]
    meses = df_entrenamiento["date_block_num"].unique()

    for i in sorted(meses):
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


# ========================================================================
# FUNCIÓN PRINCIPAL
# ========================================================================

def main(input_path, output_path):
    """
    Ejecuta el pipeline completo de preprocesamiento.
    
    Args:
        input_path: Ruta base de entrada (e.g., /opt/ml/processing/input)
        output_path: Ruta base de salida (e.g., /opt/ml/processing/output)
    """
    
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("Iniciando preprocesamiento de datos para SageMaker")
    logger.info("=" * 70)
    logger.info(f"Input path:  {input_path}")
    logger.info(f"Output path: {output_path}")
    
    # Construir rutas de archivos de entrada
    path_items = os.path.join(input_path, "items_en.csv")
    path_categories = os.path.join(input_path, "item_categories_en.csv")
    path_shops = os.path.join(input_path, "shops_en.csv")
    path_train = os.path.join(input_path, "sales_train.csv")
    path_test = os.path.join(input_path, "test.csv")
    
    # Construir rutas de salida con subdirectorios esperados por el pipeline
    path_out_train = os.path.join(output_path, "train", "datos_entreno.parquet")
    path_out_val = os.path.join(output_path, "validation", "datos_validacion.parquet")
    path_out_infer = os.path.join(output_path, "test", "datos_inferencia.parquet")
    
    # Crear directorio de salida si no existe
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # ====================================================================
        # 1. LECTURA DE DATOS
        # ====================================================================
        logger.info("\n[1/6] Cargando datasets raw...")
        
        # Verificar que todos los archivos existan
        required_files = {
            "items": path_items,
            "categories": path_categories,
            "shops": path_shops,
            "train": path_train,
            "test": path_test
        }
        
        for name, filepath in required_files.items():
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
            logger.info(f"  ✓ {name}: {filepath}")
        
        # Cargar datos
        articulos = pd.read_csv(path_items)
        categorias = pd.read_csv(path_categories)
        tiendas = pd.read_csv(path_shops)
        
        # Leer y procesar datos de entrenamiento
        datos_entrenamiento = (
            pd.read_csv(path_train)
            .merge(articulos, on="item_id", how="left")
            .merge(categorias, on="item_category_id", how="left")
            .merge(tiendas, on="shop_id", how="left")
            .assign(
                date=lambda df: pd.to_datetime(df["date"], format="%d.%m.%Y"),
                month=lambda df: df["date"].dt.month,
            )
        )
        
        logger.info(f"Datos cargados: {len(datos_entrenamiento):,} registros")
        validar_datos(datos_entrenamiento, "Datos Raw")
        
        # ====================================================================
        # 2. LIMPIEZA DE DATOS
        # ====================================================================
        logger.info("\n[2/6] Limpiando datos...")
        
        filas_antes = len(datos_entrenamiento)
        
        datos_entrenamiento = (
            datos_entrenamiento
            .query("item_price > 0")  # Eliminar precios negativos
            .query("item_price < 100000")  # Eliminar precios muy altos
            .query("item_cnt_day < 1000")  # Eliminar ventas diarias excesivas
            .drop_duplicates()
        )
        
        filas_despues = len(datos_entrenamiento)
        filas_eliminadas = filas_antes - filas_despues
        
        logger.info(f"Filas antes: {filas_antes:,}")
        logger.info(f"Filas después: {filas_despues:,}")
        logger.info(f"Filas eliminadas: {filas_eliminadas:,}")
        logger.info(f"Dimensiones: {datos_entrenamiento.shape}")
        
        # ====================================================================
        # 3. CONSOLIDACIÓN: MATRIZ MES -> TIENDA -> PRODUCTO -> VENTAS
        # ====================================================================
        logger.info("\n[3/6] Generando grid base (mes-tienda-item)...")
        
        matriz_ventas = generar_grid_base(datos_entrenamiento)
        matriz_ventas = matriz_ventas.pipe(optimizar_tipos).sort_values(
            ["date_block_num", "shop_id", "item_id"]
        )
        
        logger.info(f"Grid base generado: {matriz_ventas.shape}")
        
        # Agrupar ventas por mes
        logger.info("Agrupando ventas mensuales...")
        ventas_agrupadas = (
            datos_entrenamiento
            .groupby(["date_block_num", "shop_id", "item_id"])
            .agg(item_cnt_month=("item_cnt_day", "sum"))
            .reset_index()
        )
        
        # Unir con matriz
        cols = ["date_block_num", "shop_id", "item_id"]
        matriz_ventas = pd.merge(
            matriz_ventas, ventas_agrupadas, on=cols, how="left"
        ).assign(
            item_cnt_month=lambda df: (
                df["item_cnt_month"].fillna(0).clip(0, 20).astype(np.float32)
            )
        )
        
        logger.info(f"Matriz consolidada: {matriz_ventas.shape}")
        
        # ====================================================================
        # 4. INTEGRAR DATOS DE PRUEBA
        # ====================================================================
        logger.info("\n[4/6] Integrando datos de prueba...")
        
        datos_prueba = pd.read_csv(path_test).assign(
            date_block_num=34
        ).pipe(optimizar_tipos)
        
        logger.info(f"Datos de prueba: {datos_prueba.shape}")
        
        matriz_ventas = pd.concat(
            [matriz_ventas, datos_prueba], ignore_index=True, sort=False
        ).fillna(0)
        
        logger.info(f"Matriz con prueba: {matriz_ventas.shape}")
        
        # ====================================================================
        # 5. AGREGAR VARIABLES DE HISTORIA
        # ====================================================================
        logger.info("\n[5/6] Generando variables de historia (meses anteriores)...")
        
        matriz_ventas = matriz_ventas.pipe(
            agregar_historia, 
            meses_atras=[1, 2, 3, 12], 
            columna_base="item_cnt_month"
        ).fillna(0)
        
        cols_historia = [f"item_cnt_month_mes_ant_{m}" for m in [1, 2, 3, 12]]
        for col in cols_historia:
            if col in matriz_ventas.columns:
                matriz_ventas[col] = matriz_ventas[col].astype(np.float32)
        
        logger.info(f"Variables de historia agregadas")
        
        # Recuperar categoría
        if "ID" in matriz_ventas.columns:
            matriz_ventas = matriz_ventas.drop("ID", axis=1)
        
        matriz_ventas = pd.merge(
            matriz_ventas,
            articulos[["item_id", "item_category_id"]],
            on="item_id",
            how="left"
        )
        
        logger.info(f"Matriz final: {matriz_ventas.shape}")
        logger.info(f"Columnas: {list(matriz_ventas.columns)}")
        
        # ====================================================================
        # 6. GUARDAR DATASETS
        # ====================================================================
        logger.info("\n[6/6] Guardando datasets procesados...")
        
        df_train = matriz_ventas[matriz_ventas.date_block_num < 33]
        df_val = matriz_ventas[matriz_ventas.date_block_num == 33]
        df_infer = matriz_ventas[matriz_ventas.date_block_num == 34]
        
        guardar_dataset(df_train, path_out_train)
        guardar_dataset(df_val, path_out_val)
        guardar_dataset(df_infer, path_out_infer)
        
        # Resumen final
        duration = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("✓ PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 70)
        logger.info(f"Tiempo de ejecución: {duration:.2f} segundos")
        logger.info(f"\nDatasets generados:")
        logger.info(f"  - Entrenamiento: {df_train.shape} → {path_out_train}")
        logger.info(f"  - Validación:    {df_val.shape}   → {path_out_val}")
        logger.info(f"  - Inferencia:    {df_infer.shape}   → {path_out_infer}")
        
    except Exception as e:
        logger.error("=" * 70)
        logger.error("✗ ERROR CRÍTICO EN EL PREPROCESAMIENTO")
        logger.error("=" * 70)
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesamiento de datos para SageMaker Processing Job"
    )
    
    parser.add_argument(
        "--input-path",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Ruta de entrada (/opt/ml/processing/input)"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Ruta de salida (/opt/ml/processing/output)"
    )
    
    args = parser.parse_args()
    
    main(args.input_path, args.output_path)