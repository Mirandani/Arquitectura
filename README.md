# ITAM Maestría en Ciencia de Datos

## Métodos de Gran Escala


**Blanca Azucena Orduña López**

**Daniel Miranda Badillo*

Repositorio de proyectos del curso Métodos de Gran Escala 

https://github.com/Mirandani/Arquitectura

### Estructura del repositorio

```
.
├── README.md
├── artifacts                             # documentación y soporte                           
│   ├── documentation 
│   │   ├── Executive Summary.pdf
│   │   └── image.png
│   └── models
│       └── modelo_random_forest.joblib
├── data                                                
│   ├── inference                         # datos para predicciones en batch
│   │   └── datos_inferencia.parquet
│   ├── predictions
│   │   ├── predicciones_batch.csv        # predicciones con datos entrenados
│   │   └── submission.csv
│   ├── prep                              # datos preparados para el modelado
│   │   ├── datos_entreno.parquet
│   │   └── datos_validacion.parquet
│   └── raw                               # datos originales
│       ├── item_categories.csv
│       ├── item_categories_en.csv
│       ├── items.csv
│       ├── items_en.csv
│       ├── sales_train.csv
│       ├── shops.csv
│       ├── shops_en.csv
│       ├── submission.csv
│       └── test.csv
├── data copy
├── demo-boto3
│   ├── README.md
│   ├── main.py
│   ├── pyproject.toml
│   └── uv.lock
├── main.py
├── notebooks                           # código de tratamiento y modelado de datos
│   ├── modelo_retail.ipynb             
│   └── notebook.ipynb
├── notes
│   └── notas.md
├── pyproject.toml
├── src                                 # archivos .py
│   ├── __init__.py
│   ├── inference.py
│   ├── prep.py
│   └── train.py
├── tarea_01
│   ├── experimentos
│   │   ├── EDA2_retail.ipynb
│   │   └── EDA_retail.ipynb
│   ├── pyproject.toml
│   └── uv.lock
└── uv.lock

```