"""_summary_
La entrada del script son datos data/raw.
La salida del script son datos data/prep.
Este modulo sirve para hacer las transformaciones necesarias para dejar
los datos listos para el analisis exploratorio y modelado.
"""

from itertools import product
import pandas as pd
import numpy as np

# Funciones

# función para generar la matriz base de mes-tienda-item
def generar_grid_base(df_entrenamiento):
    """Genera todas las combinaciones posibles de mes-tienda-item."""
    grid = []
    cols = ['date_block_num', 'shop_id', 'item_id']
    
    # Usamos el máximo mes en los datos + iteramos
    meses = df_entrenamiento['date_block_num'].unique()
    
    for i in meses:
        ventas = df_entrenamiento[df_entrenamiento.date_block_num == i]
        grid.append(np.array(list(product([i], ventas.shop_id.unique(), ventas.item_id.unique())), dtype='int16'))
    
    return pd.DataFrame(np.vstack(grid), columns=cols)


# función para cambiar tipos de datos
def optimizar_tipos(df):
    """Reduce el uso de memoria casteando IDs a enteros pequeños."""
    return df.assign(
        date_block_num=lambda x: x['date_block_num'].astype(np.int8),
        shop_id=lambda x: x['shop_id'].astype(np.int8),
        item_id=lambda x: x['item_id'].astype(np.int16)
    )

# función para guardado de datasets parquet
def guardar_dataset(df, ruta):
    """Guarda el dataframe en parquet e imprime confirmación."""
    df.to_parquet(ruta, index=False)
    print(f"Guardado: {ruta}  Dimensiones: {df.shape}")

# función para agregar historia de ventas
def agregar_historia(datos, meses_atras, columna_base):
    df_temp = datos[['date_block_num', 'shop_id', 'item_id', columna_base]]
    
    for mes in meses_atras:
        desplazado = df_temp.copy()
        desplazado.columns = ['date_block_num', 'shop_id', 'item_id', f"{columna_base}_mes_ant_{mes}"]
        desplazado['date_block_num'] += mes
        
        datos = pd.merge(datos, desplazado, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return datos

pd.set_option('display.float_format', lambda x: '%.2f' % x)

###########################################################################
# CARGA DE DATOS
###########################################################################

# Lectura de datos
datos_entrenamiento = pd.read_csv('data/raw/sales_train.csv')
articulos = pd.read_csv('data/raw/items_en.csv') # traducción de nombres de artículos a inglés
categorias = pd.read_csv('data/raw/item_categories_en.csv') # traducción de categorías a inglés
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

# Verificación del número de filas y columnas
print("filas y columnas datos_entrenamiento",datos_entrenamiento.shape)
print(datos_entrenamiento.head())
print(datos_entrenamiento.describe())

# Conteo nulos
print("Valores nulos")
nulos = datos_entrenamiento.isnull().sum()
print(nulos[nulos > 0])


# REVISIÓN INCIAL DE VENTAS MENSUALES

# Agrupar ventas por mes
ventas_mensuales = datos_entrenamiento.groupby('date_block_num')['item_cnt_day'].sum()

# Gráficas
"""
plt.figure(figsize=(12, 5))
plt.plot(ventas_mensuales.index, ventas_mensuales.values, marker='o', color='b')
plt.title('Ventas ene 2013 - oct 2015')
plt.xlabel('Mes 0 = ene 2013')
plt.ylabel('Artículos vendidos')
plt.axvline(x=11, color='r', linestyle='--', label='dic 2013')
plt.axvline(x=23, color='r', linestyle='--', label='dic 2014')
plt.legend()
plt.show()
"""


# REVISIÓN OUTLIERS
# Para número de artículos vendidos y comportamiento de precios
#fig, ax = plt.subplots(1, 2, figsize=(14, 4))

#sns.boxplot(x=datos_entrenamiento['item_cnt_day'], ax=ax[0])
#ax[0].set_title('articulos vendidos por día')

#sns.boxplot(x=datos_entrenamiento['item_price'], ax=ax[1])
#ax[1].set_title('Precios')

#plt.show()


# Revisión de precios
#datos_entrenamiento.sort_values("item_price", ascending=False).head(15)

###########################################################################
# LIMPIEZA DE DATOS
###########################################################################
print("Limpieza de datos...")

datos_entrenamiento = (
    datos_entrenamiento
    .query('item_price > 0')           # Eliminar precios negativos
    .query('item_price < 100000')      # Eliminar precios muy altos
    .query('item_cnt_day < 1000')      # Eliminar ventas diarias excesivas
    .drop_duplicates()
)

print("Filas y columnas después de limpieza", datos_entrenamiento.shape)


##############################################################
#  analisis exploratorio
##############################################################

# Revisión de tiendas abiertas
ventas_por_tienda = (
    datos_entrenamiento
    .pivot_table(index='shop_id', columns='date_block_num', values='item_cnt_day', aggfunc='sum')
    .fillna(0)
)

# Meses sin ventas  rellenar nas  con 0
ventas_por_tienda = ventas_por_tienda.fillna(0)

# Heat map
#plt.figure(figsize=(20, 10))
#sns.heatmap(ventas_por_tienda, cmap='viridis', vmin=0, vmax=2000) # vmin/vmax ajustan el contraste
#plt.title('Ventas por tienda')
#plt.xlabel('mes 0 = ene 2013')
#plt.ylabel('ID Tienda')
#plt.show()


# Tiendas cerradas
tiendas_cerradas = ventas_por_tienda.iloc[:, -2:][lambda df: df.sum(axis=1) == 0]
print(f"{len(tiendas_cerradas)} tiendas sin ventas los últimos 2 meses")


# Check 2 ventas por mes
datos_entrenamiento['month'] = datos_entrenamiento['date'].dt.month

# Suma de ventas por mes
ventas_mensuales = datos_entrenamiento.groupby(['date_block_num', 'month'])['item_cnt_day'].sum().reset_index()

#plt.figure(figsize=(12, 6))
#sns.boxplot(x='month', y='item_cnt_day', data=ventas_mensuales)
#plt.title('Distribución ventas por mes')
#plt.xlabel('Mes (1=ene , 12= dic)')
#plt.ylabel('Total de Ventas')
#plt.show()


# CATEGORÍAS TOP

# Ventas por categoría
cat_ventas = (
    datos_entrenamiento
    .groupby('item_category_name')['item_cnt_day']
    .sum()
    .sort_values(ascending=False)
)

# Top 20
#plt.figure(figsize=(12, 8))
#sns.barplot(y=cat_ventas.index[:20], x=cat_ventas.values[:20], palette='magma')
#plt.title('Top 20 categorías más vendidas')
#plt.xlabel('# unidades vendidas')
#plt.show()


############################
# plot ventas por item_category_name vs mes

# agrupar por mes, tienda, artículo
datos_entrenamiento_agrupados = (
    datos_entrenamiento
    .groupby(['date_block_num', 'shop_id', 'item_id'])
    .agg({
        'item_cnt_day': 'sum',
        'item_price': 'mean',
        "item_category_id": 'first',
        "item_category_name": 'first',
        "shop_name": 'first'
    })
    .reset_index()
)

#plt.figure(figsize=(15, 7))
#sns.lineplot(data=datos_entrenamiento_agrupados, x='date_block_num', y='item_cnt_day', hue='item_category_name', estimator='sum', ci=None)
#plt.title('Ventas mensuales por categoría de artículo')
#plt.xlabel('Mes (0 = ene 2013)')
#plt.ylabel('Número de artículos vendidos')
#plt.legend(title='Categoría de artículo', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.show()

#plot ventas por shop_name vs mes
#plt.figure(figsize=(15, 7))
#sns.lineplot(data=datos_entrenamiento_agrupados, x='date_block_num', y='item_cnt_day', hue='shop_name', estimator='sum', ci=None)
#plt.title('Ventas mensuales por tienda')
#plt.xlabel('Mes (0 = ene 2013)')
#plt.ylabel('Número de artículos vendidos')
#plt.legend(title='Tienda', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.show()

# Revisión de tiendas datos_entrenamiento vs datos_prueba
datos_prueba = pd.read_csv('data/raw/test.csv')

# Tiendas
tiendas_entrenamiento = set(datos_entrenamiento['shop_id'].unique())
tiendas_prueba = set(datos_prueba['shop_id'].unique())
print("Tiendas nuevas en datos_prueba:", tiendas_prueba - tiendas_entrenamiento)

# Productos
art_entrenamiento = set(datos_entrenamiento['item_id'].unique())
art_prueba = set(datos_prueba['item_id'].unique())
print("Productos nuevos en datos_prueba:", len(art_prueba - art_entrenamiento))


###########################################################################
# CONSOLIDACIÓN DE INFORMACIÓN MES-> TIENDA -> PRODUCTO -> VENTAS
###########################################################################


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
        item_cnt_month=lambda df: df['item_cnt_month'].fillna(0).clip(0, 20).astype(np.float16)
    )
)

print("Dimensiones matriz_ventas:", matriz_ventas.shape)
#matriz_ventas.head()


# Filas con 0 ventas
total_filas = matriz_ventas.shape[0]
cnt_ventas_cero = matriz_ventas.query('item_cnt_month == 0').shape[0]
print(f"Porcentaje sin ventas: {(cnt_ventas_cero / total_filas) * 100:.2f}%")

###########################################################################
#.   MATRIZ MES TIENDA PRODUCTO + DATOS_PRUEBA + HISTORIA
###########################################################################

# Preparar datos_prueba para unión con matriz
datos_prueba = (
    pd.read_csv('data/raw/test.csv')
    .assign(date_block_num=34)
    .pipe(optimizar_tipos)
)

# Unión de datos_prueba con matriz consolidada
matriz_ventas = pd.concat([matriz_ventas, datos_prueba], ignore_index=True, sort=False).fillna(0)

#  GENERACIÓN DE DATASETS ABT
# variables con número de mes
matriz_ventas = (
    matriz_ventas
    .pipe(agregar_historia, meses_atras=[1, 2, 3, 12], columna_base='item_cnt_month')
    .fillna(0)
)

##### GUARDANDO DATASETS

# División de datos
print("Generando datasets de entrenamiento, validación y prueba final...")
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

print("\nProceso finalizado con éxito.")


