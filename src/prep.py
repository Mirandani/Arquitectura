"""_summary_
La entrada del script son datos data/raw. La salida del script son datos data/prep.
Este modulo sirve para hacer las transformaciones necesarias para dejar los datos listos para el analisis exploratorio y modelado.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# Lectura de datos
train = pd.read_csv('data/raw/sales_train.csv')
items = pd.read_csv('data/raw/items_en.csv') 
cats = pd.read_csv('data/raw/item_categories_en.csv')
shops = pd.read_csv('data/raw/shops_en.csv')

# Join de dataframes
train = pd.merge(train, items, on='item_id', how='left')

# train + categorías
train = pd.merge(train, cats, on='item_category_id', how='left')

# # train + shop
train = pd.merge(train, shops, on='shop_id', how='left')

# Formato fecha
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')

# Verificación
print("filas y columnas train",train.shape)
train.head()

pd.set_option('display.float_format', lambda x: '%.2f' % x)
train.describe()

# Conteo nulos 
print("Valores nulos")
nulos = train.isnull().sum()
print(nulos[nulos > 0]) 


# REVISIÓN INCIAL DE VENTAS MENSUALES

# Agrupar ventas por mes 
monthly_sales = train.groupby('date_block_num')['item_cnt_day'].sum()

# Gráficas 
"""
plt.figure(figsize=(12, 5))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', color='b')
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

#sns.boxplot(x=train['item_cnt_day'], ax=ax[0])
#ax[0].set_title('Items vendidos por día')

#sns.boxplot(x=train['item_price'], ax=ax[1])
#ax[1].set_title('Precios')

#plt.show()


# Revisión de precios
#train.sort_values("item_price", ascending=False).head(15)

###############################
# Limpieza valores atípicos

# Eliminar precios negativos
train = train[train['item_price'] > 0]

# Eliminar precios muy altos
train = train[train['item_price'] < 100000]

#Eliminar número de unidades vendidas muy altas
train = train[train['item_cnt_day'] < 1000]


## Devoluciones 
#print(" % devoluciones:", (train[train['item_cnt_day'] < 0].shape[0] / train.shape[0]) * 100)

# Duplicados 

print("Duplicados:", train.duplicated().sum())
train = train.drop_duplicates()

print("Filas y columnas después de limpieza", train.shape)



###########################
# Revisión de tiendas abiertas 
sales_by_shop = train.pivot_table(index='shop_id', columns='date_block_num', values='item_cnt_day', aggfunc='sum')

# Meses sin ventas  rellenar nas  con 0
sales_by_shop = sales_by_shop.fillna(0)

# Heat map
#plt.figure(figsize=(20, 10))
#sns.heatmap(sales_by_shop, cmap='viridis', vmin=0, vmax=2000) # vmin/vmax ajustan el contraste
#plt.title('Ventas por tienda')
#plt.xlabel('mes 0 = ene 2013')
#plt.ylabel('ID Tienda')
#plt.show()

# Tiendas sin ventas los últimos 2 meses
ultimos_meses = sales_by_shop.iloc[:, -2:]

# Tiendas venta 0
tiendas_cerradas = ultimos_meses[ultimos_meses.sum(axis=1) == 0]

print(f"{len(tiendas_cerradas)} tiendas sin ventas los últimos 2 meses")




# Check 2 ventas por mes 

train['month'] = train['date'].dt.month

# Suma de ventas por mes
monthly_sales = train.groupby(['date_block_num', 'month'])['item_cnt_day'].sum().reset_index()

#plt.figure(figsize=(12, 6))
#sns.boxplot(x='month', y='item_cnt_day', data=monthly_sales)
#plt.title('Distribución ventas por mes')
#plt.xlabel('Mes (1=ene , 12= dic)')
#plt.ylabel('Total de Ventas')
#plt.show()


# CATEGORÍAS TOP

# Ventas por categoría
cat_sales = train.groupby('item_category_name')['item_cnt_day'].sum().sort_values(ascending=False)

# Top 20
#plt.figure(figsize=(12, 8))
#sns.barplot(y=cat_sales.index[:20], x=cat_sales.values[:20], palette='magma')
#plt.title('Top 20 categorías más vendidas')
#plt.xlabel('# unidades vendidas')
#plt.show()


############################
# plot ventas por item_category_name vs mes

# agrupar por mes, tienda, artículo
train_gpd = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({
    'item_cnt_day': 'sum',
    'item_price': 'mean',
    "item_category_id": 'first',
    "item_category_name": 'first',
    "shop_name":"first"
}).reset_index()

#plt.figure(figsize=(15, 7))
#sns.lineplot(data=train_gpd, x='date_block_num', y='item_cnt_day', hue='item_category_name', estimator='sum', ci=None)
#plt.title('Ventas mensuales por categoría de artículo')
#plt.xlabel('Mes (0 = ene 2013)')
#plt.ylabel('Número de artículos vendidos')
#plt.legend(title='Categoría de artículo', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.show()  

#plot ventas por shop_name vs mes
#plt.figure(figsize=(15, 7))
#sns.lineplot(data=train_gpd, x='date_block_num', y='item_cnt_day', hue='shop_name', estimator='sum', ci=None)
#plt.title('Ventas mensuales por tienda')
#plt.xlabel('Mes (0 = ene 2013)')
#plt.ylabel('Número de artículos vendidos')
#plt.legend(title='Tienda', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.show()  

# Revisión de tiendas train vs test
test = pd.read_csv('data/raw/test.csv')

# Tiendas 
tiendas_train = set(train['shop_id'].unique())
tiendas_test = set(test['shop_id'].unique())
print("Tiendas nuevas en test:", tiendas_test - tiendas_train)

# Productos 
items_train = set(train['item_id'].unique())
items_test = set(test['item_id'].unique())
print("Productos nuevos en test:", len(items_test - items_train))


###############
# CONSOLIDACIÓN DE INFORMACIÓN MES-> TIENDA -> PRODUCTO -> VENTAS
from itertools import product

# Matriz mes-tienda-item
matrix = []
cols = ['date_block_num','shop_id','item_id']

for i in range(34):
    sales = train[train.date_block_num==i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols, inplace=True)

# Incluimos las ventas por mes
group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=cols, how='left')

# Reemplazamos nulos por 0
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0,20) # Limitamos a 20 como pide la competencia
                                .astype(np.float16))

print("Dimensiones matriz:", matrix.shape)
#matrix.head()


# Filas con 0 ventas
ceros = matrix[matrix['item_cnt_month'] == 0].shape[0]
total = matrix.shape[0]
porcentaje_sin_ventas = (ceros / total) * 100

print(f"Total de combinaciones Mes-Tienda-Producto: {total}")
print(f"Combinaciones con 0 ventas: {ceros}")
print(f"Porcentaje sin ventas: {porcentaje_sin_ventas:.2f}%")



# Cargar test.csv
test = pd.read_csv('data/raw/test.csv')
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)

# Unión de test con matriz consolidada
matriz = pd.concat([matrix, test], ignore_index=True, sort=False)
matriz.fillna(0, inplace=True)

# Función historia, crear variable de meses anteriores
def historia(datos, meses_atras, columna_base):
    temp = datos[['date_block_num','shop_id','item_id', columna_base]]
    
    for i in meses_atras:
        desplazado = temp.copy()
        nombre_columna = f"{columna_base}_mes_ant_{i}"
        desplazado.columns = ['date_block_num','shop_id','item_id', nombre_columna]
        
        desplazado['date_block_num'] += i
        
        datos = pd.merge(datos, desplazado, on=['date_block_num','shop_id','item_id'], how='left')
    return datos

#  GENERACIÓN DE DATASETS ABT
# variables con número de mes
matriz = historia(matriz, [1, 2, 3, 12], 'item_cnt_month')
matriz = matriz.fillna(0)

######### 
# División de datos
print("Generando datasets de entrenamiento, validación y prueba final...")
datos_entreno = matriz[matriz.date_block_num < 33]
datos_validacion = matriz[matriz.date_block_num == 33]
datos_prueba_final = matriz[matriz.date_block_num == 34]

# La salida del script son datos data/prep 
datos_entreno.to_csv('data/prep/datos_entreno.csv', index=False)
datos_validacion.to_csv('data/prep/datos_validacion.csv', index=False)
datos_prueba_final.to_csv('data/prep/datos_prueba_final.csv', index=False)

