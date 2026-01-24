#  ITAM Maestría en Ciencia de Datos

## Tarea 1: Predicción de Demanda en Retail con Machine Learning



**Blanca Azucena Orduña López**  
**Daniel Miranda Badillo**

**Dataset:** Retail Sales Forecasting Dataset (Kaggle)

Se presenta un caso de estudio para el análisis de datos de la compañía rusa **1C Company**  que opera una cadena de tiendas en múltiples ciudades.
La compañía tiene el reto de modernizar su proceso de inventarios para eficientar los costos operativos, de almacén y maximizar sus operaciones.

## Executive Summary
https://github.com/Mirandani/Arquitectura/blob/main/tarea_01/Executive%20Summary.pdf

## Estructura del Proyecto
```
tarea_01/
├── modelo_retail.ipynb       # Notebook con EDA, Modelado y predicción de demanda 
├── Executive Sumary.pdf      # Resumen ejecutivo para stakeholders (https://github.com/Mirandani/Arquitectura/blob/main/tarea_01/Executive%20Summary.pdf)
├── README.md                 # Documentación del proyecto                   
│── data/                    # Carpeta con datasets
    ├── sales_train.csv       # Datos históricos de ventas
    ├── items.csv             # Información de los productos
    ├── item_categories.csv   # Categorías de los productos
    ├── shops.csv             # Información de las tiendas
    └── test.csv              # Datos para realizar las predicciones
├── pyproject.toml            # Archivo de configuración de uv
├── uv.lock                   # Archivo de bloqueo de dependencias de uv
```

## Ejecución del Proyecto con uv
```
git clone git@github.com:Mirandani/Arquitectura.git
cd Arquitectura
cd tarea_01
uv sync
uv run jupyter lab
```
## Resumen
Se evaluaron dos modelos de machine learning: **regresión lineal  y random forest** para el prónostico de inventarios buscando obtener un RMSE < 5 unidades.

## Relevantes

- **Modelo ganador**: Random forest
- **RMSE obtenidio**: 0.9743 unidades

## Kaggle
Score: 1.02184
![alt text](image.png)

## Contenido

1. modelo_retail.ipynb
  - Análisis exploratorio de datos y limpieza
  - Ingeniería de variables, modelado y predicción
2. Executive Sumary.pdf
  - Resumen ejecutivo para stakeholders


