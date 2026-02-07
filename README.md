![alt text](image.png)

# ITAM Maestría en Ciencia de Datos

## Métodos de Gran Escala

### Autores
- **Blanca Azucena Orduña López**
- **Daniel Miranda Badillo**

**Repositorio:** https://github.com/Mirandani/Arquitectura

## Descripción del proyecto
Este repositorio contiene el código, datos y documentación relacionados con el proyecto de Métodos de Gran Escala. El objetivo del proyecto es desarrollar un modelo de machine learning para predecir ventas en un contexto de retail, utilizando técnicas de preparación de datos, modelado y evaluación.

## Datos utilizados
Los datos provienen de la competencia de Kaggle **"Predict Future Sales"**. El dataset incluye información sobre ventas históricas, productos, tiendas y fechas.

- **Dataset:** https://www.kaggle.com/c/competitive-data-science-predict-future-sales
- **Leaderboard:** https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/leaderboard

## Resultados
**Score en Kaggle:** 1.02

**Métricas de entrenamiento:**
- RMSE Regresión Lineal: 0.9824
- RMSE Random Forest: 0.9743

**Estadísticas de predicciones:**
- Media: 0.28
- Mediana: 0.14
- Mínimo: 0.05
- Máximo: 19.43

### Estructura del repositorio


```
.
├── .gitignore                            # archivos ignorados por git
├── README.md
├── Makefile                              # comandos de desarrollo (lint, format, tree)
├── main.py
├── pyproject.toml
├── uv.lock
├── artifacts                             # documentación y modelos
│   ├── documentation
│   │   ├── Executive_Summary.pdf
│   │── logs                              # logs de ejecución de scripts
│   │   ├── prep.log
│   └── models
│       └── modelo_random_forest.joblib
├── data                                  # datos del proyecto
│   ├── inference                         # datos para predicciones en batch
│   ├── predictions                       # resultados de predicciones
│   ├── prep                              # datos preparados para modelado
│   └── raw                               # datos originales sin procesar
├── demo-boto3                            # demo de integración con AWS
│   ├── README.md
│   ├── main.py
│   ├── pyproject.toml
│   └── uv.lock
├── notebooks                             # notebooks de exploración y modelado
│   ├── modelo_retail.ipynb
│   └── notebook.ipynb
├── notes                                 # documentación y notas
│   ├── buenas_practicas.md
│   └── notas.md
├── src                                   # código fuente del proyecto
│   ├── __init__.py
│   ├── file_checker.py                   # verificación de archivos
│   ├── inference.py                      # predicciones con modelo entrenado
│   ├── prep.py                           # preparación de datos
│   ├── train.py                          # entrenamiento de modelos
│   └── utils                             # utilidades compartidas
│       ├── __init__.py
│       ├── data_validation.py            # validación de datos
│       ├── dtypes.py                     # manejo de tipos de datos
│       ├── logger.py                     # configuración de logging
│       ├── model_tools.py                # herramientas de modelado
│       └── outputs.py                    # funciones de salida de datos
└── tarea_01                              # trabajos y experimentos
    ├── data
    ├── experimentos
    │   ├── EDA_retail.ipynb
    │   └── EDA2_retail.ipynb
    ├── pyproject.toml
    └── uv.lock
```

## Requerimientos
- Python 3.11 o superior
- [uv](https://github.com/astral-sh/uv) para gestión de dependencias

## Instalación y configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/Mirandani/Arquitectura.git
cd Arquitectura
```

### 2. Instalar dependencias
```bash
uv sync
```

## Pipeline de ejecución

### 1. Preparación de datos
Carga datos originales, realiza limpieza, ingeniería de características y guarda datos preparados en formato parquet.
```bash
uv run python src/prep.py
```

### 2. Entrenamiento del modelo
Entrena modelos de machine learning (Regresión Lineal y Random Forest) y guarda el mejor modelo.
```bash
uv run python src/train.py
```

### 3. Inferencia
Carga el modelo entrenado, genera predicciones y guarda los resultados.
```bash
uv run python src/inference.py
```


## Comandos de desarrollo (Makefile)

El proyecto incluye un Makefile con comandos útiles para desarrollo:

### Análisis de código
```bash
make lint              # Ejecuta pylint y muestra resultados en terminal
make lint-report       # Ejecuta pylint y guarda reporte en pylint_report.txt
```

### Formateo de código
```bash
make format-black      # Formatea código con black
make format-black-check # Verifica formato sin modificar archivos
make format-black-list  # Lista archivos que necesitan formato
make format-ruff        # Formatea código con ruff
make format-ruff-check  # Verifica formato con ruff
```

### Utilidades
```bash
make tree              # Muestra estructura del proyecto
make help              # Muestra todos los comandos disponibles
```

