![alt text](image.png)

# ITAM Maestría en Ciencia de Datos — Métodos de Gran Escala

## Autores

- **Blanca Azucena Orduña López**
- **Daniel Miranda Badillo**

**Repositorio:** <https://github.com/Mirandani/Arquitectura>

---

## Descripción del Proyecto

Este repositorio implementa un **pipeline completo de Machine Learning** para predecir ventas en retail usando técnicas de preparación de datos, entrenamiento de modelos y inferencia en batch.

### Caso de uso
Predecir ventas futuras de productos en diferentes tiendas basándose en datos históricos.

### Objetivo
Desarrollar un modelo escalable, testeable y containerizado que permita:
- **Preparación de datos** con validación automática
- **Entrenamiento flexible** con múltiples modelos y hiperparámetros configurables
- **Inferencia en batch** sobre nuevos datos
- **Calidad de código** mediante linting, formateo y pruebas unitarias

### Datos
- **Fuente:** Kaggle — [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales)
- **Métricas de desempeño:**
  - RMSE Regresión Lineal (baseline): 0.9824
  - RMSE Random Forest (principal): 0.9743
  - **Mejora:** ~0.83% respecto al baseline

---

## Estructura del Repositorio

```
.
├── README.md                            # Este archivo
├── Makefile                             # Tareas de desarrollo y Docker
├── pyproject.toml                       # Configuración del proyecto (uv + pytest)
├── uv.lock                              # Dependencias fijadas
│
├── data/                                # Datos del pipeline
│   ├── raw/                             # Datos originales (sin procesar)
│   ├── prep/                            # Datos preparados (parquet)
│   ├── inference/                       # Datos para predicciones en batch
│   └── predictions/                     # Resultados de inferencia
│
├── artifacts/                           # Modelos entrenados y documentación
│   ├── models/
│   │   └── modelo_random_forest.joblib
│   ├── logs/                            # Logs de ejecución
│   └── documentation/
│       └── Executive_Summary.pdf
│
├── src/                                 # Código fuente modular
│   ├── utils/                           # Utilidades compartidas
│   │   ├── __init__.py
│   │   ├── logger.py                    # Configuración de logging
│   │   ├── model_tools.py               # Evaluación y serialización de modelos
│   │   ├── data_validation.py           # Validación de datos
│   │   ├── dtypes.py                    # Gestión de tipos de datos
│   │   └── outputs.py                   # Exportación de resultados
│   │
│   ├── preprocessing/                   # Módulo de preprocesamiento
│   │   ├── __init__.py
│   │   ├── Dockerfile                   # Imagen Docker para preprocessing
│   │   ├── __main__.py                  # CLI con argparse
│   │   └── preprocessing.py             # Funciones puras de transformación
│   │
│   ├── training/                        # Módulo de entrenamiento
│   │   ├── __init__.py
│   │   ├── Dockerfile                   # Imagen Docker para entrenamiento
│   │   ├── __main__.py                  # CLI con argparse
│   │   ├── train.py                     # Funciones puras de modelado
│   │   └── test/
│   │       ├── __init__.py
│   │       └── test_train.py            # Tests unitarios
│   │
│   └── inference/                       # Módulo de predicciones
│       ├── __init__.py
│       ├── Dockerfile                   # Imagen Docker para inferencia
│       ├── __main__.py                  # CLI con argparse
│       ├── inference.py                 # Funciones puras de predicción
│       └── test/
│           ├── __init__.py
│           └── test_inference.py        # Tests unitarios
│
├── notebooks/                           # Exploración y análisis
│   └── modelo_retail.ipynb
│
└── notes/                               # Documentación interna
    ├── buenas_practicas.md
    └── notas.md
```

---

## Git Workflow

![alt text](image-8.png)

### Rama principal: `main`
- Código estable y testeado
- Se actualiza solo mediante Pull Requests

### Ramas de desarrollo: 
- `development` Integración (rama activa de desarrollo)
- `feature`.    Integración (rama activa de desarrollo)

```bash
git checkout -b feature/nombre-feature    # Rama para nuevas funcionalidades
git checkout -b fix/nombre-bug            # Rama para correcciones
git checkout development                  # Rama de desarrollo
git pull origin devlopment                # Actualización de la rama

```

### Proceso de integración
1. Crear rama desde `main`
2. Realizar cambios y commits
3. Crear Pull Request con descripción clara
4. Pasar tests y revisión de código
5. Merge a `main`

---

## Instalación y Setup

### Requisitos previos
- **Python:** 3.12+
- **Docker:** (opcional, para ejecución containerizada)
- **Git:** para clonar el repositorio

### 1. Clonar el repositorio

```bash
git clone https://github.com/Mirandani/Arquitectura.git
cd Arquitectura
```

### 2. Crear e instalar el ambiente virtual

```bash
# Instalar dependencias con uv
uv sync

# Activar el ambiente (opcional, uv run lo hace automáticamente)
source .venv/bin/activate  # en macOS/Linux
```

### 3. Verificar instalación

```bash
uv run python -c "import pandas; print('✓ Dependencias instaladas')"
```

---

## Ejecución del Pipeline Completo

### Opción A: Sin Docker (local)

```bash
# 0. Preprocesamiento de datos (limpieza, ingeniería de características)
uv run python -m preprocessing

# 1. Entrenamiento con defaults
uv run python -m training

# 2. Inferencia con defaults
uv run python -m inference
```

### Opción B: Con Docker (contenedores aislados)

#### Paso 0: Preprocesamiento de datos

```bash
make docker-build-prep
make docker-run-prep
```

#### Paso 1: Entrenamiento con defaults

```bash
make docker-build-train
make docker-run-train
```

#### Paso 2: Inferencia con defaults

```bash
make docker-build-inference
make docker-run-inference
```

---

## Ejecución de Contenedores con Argumentos

### Preprocessing — Rutas de Datos Configurables

**Estructura base:**
```bash
make docker-run-prep \
  ITEMS_PATH=data/raw/items_en.csv \
  CATEGORIES_PATH=data/raw/item_categories_en.csv \
  SHOPS_PATH=data/raw/shops_en.csv \
  TRAIN_PATH=data/raw/sales_train.csv \
  TEST_PATH=data/raw/test.csv \
  OUT_TRAIN=data/prep/datos_entreno.parquet \
  OUT_VAL=data/prep/datos_validacion.parquet \
  OUT_INFER=data/inference/datos_inferencia.parquet
```

**Ejemplos:**

```bash
# Preprocesamiento con rutas customizadas
make docker-run-prep \
  TRAIN_PATH=data/raw/sales_train_full.csv \
  OUT_TRAIN=data/prep/entreno_v2.parquet

# Cambiar solo rutas de salida
make docker-run-prep \
  OUT_TRAIN=data/prep/entreno_completo.parquet \
  OUT_VAL=data/prep/validacion_completo.parquet
```

### Training — Modelos y Hiperparámetros Configurables

**Estructura base:**
```bash
make docker-run-train \
  MODELO_BASE=linear \
  MODELO_PRINCIPAL=random_forest \
  N_ESTIMATORS=200 \
  MAX_DEPTH=6 \
  RANDOM_STATE=42
```

**Ejemplos:**

```bash
# Random Forest con hiperparámetros personalizados
make docker-run-train \
  MODELO_PRINCIPAL=random_forest \
  N_ESTIMATORS=200 \
  MAX_DEPTH=8

# Random Forest vs Random Forest (comparar configuraciones)
make docker-run-train \
  MODELO_BASE=random_forest \
  MODELO_PRINCIPAL=random_forest \
  RANDOM_STATE=123

# Lineal vs Lineal (baseline puro)
make docker-run-train \
  MODELO_BASE=linear \
  MODELO_PRINCIPAL=linear
```

### Inference — Datos y Rutas Personalizadas

**Estructura base:**
```bash
make docker-run-inference \
  DATOS=data/inference/datos_inferencia.parquet \
  MODELO=artifacts/models/modelo_random_forest.joblib \
  SALIDA=data/predictions/predicciones_batch.csv
```

**Ejemplos:**

```bash
# Inferencia con datos alternativos
make docker-run-inference \
  DATOS=data/inference/datos_nuevos.parquet

# Cambiar ruta de salida
make docker-run-inference \
  SALIDA=data/predictions/predicciones_v2.csv
```

### Variables de configuración en el Makefile

#### Preprocessing
| Variable | Default | Descripción |
|----------|---------|-------------|
| `ITEMS_PATH` | `data/raw/items_en.csv` | Catálogo de productos |
| `CATEGORIES_PATH` | `data/raw/item_categories_en.csv` | Categorías de productos |
| `SHOPS_PATH` | `data/raw/shops_en.csv` | Información de tiendas |
| `TRAIN_PATH` | `data/raw/sales_train.csv` | Datos históricos de ventas |
| `TEST_PATH` | `data/raw/test.csv` | Datos de prueba para inferencia |
| `OUT_TRAIN` | `data/prep/datos_entreno.parquet` | Salida: datos de entrenamiento |
| `OUT_VAL` | `data/prep/datos_validacion.parquet` | Salida: datos de validación |
| `OUT_INFER` | `data/inference/datos_inferencia.parquet` | Salida: datos de inferencia |

#### Training
| Variable | Default | Descripción |
|----------|---------|-------------|
| `MODELO_BASE` | `linear` | Modelo baseline para comparación |
| `MODELO_PRINCIPAL` | `random_forest` | Modelo principal (se guarda) |
| `N_ESTIMATORS` | 50 | Número de árboles (RandomForest) |
| `MAX_DEPTH` | 10 | Profundidad máxima (RandomForest) |
| `RANDOM_STATE` | 42 | Semilla de reproducibilidad |
| `ENTRADA` | `data/prep/datos_entreno.parquet` | Ruta datos entrenamiento |
| `VALIDACION` | `data/prep/datos_validacion.parquet` | Ruta datos validación |
| `SALIDA_TRAIN` | `artifacts/models/modelo_random_forest.joblib` | Ruta modelo guardado |

#### Inference
| Variable | Default | Descripción |
|----------|---------|-------------|
| `DATOS` | `data/inference/datos_inferencia.parquet` | Ruta datos inferencia |
| `SALIDA` | `data/predictions/predicciones_batch.csv` | Ruta predicciones |

---

## Pipeline Completo: Ejecución Rápida

Ejecuta todos los pasos en orden (preprocessing → training → inference):

```bash
# Con valores por defecto
make docker-build-prep docker-build-train docker-build-inference && \
make docker-run-prep && \
make docker-run-train && \
make docker-run-inference

# Con parámetros personalizados
make docker-build-prep docker-build-train docker-build-inference && \
make docker-run-prep && \
make docker-run-train \
  MODELO_PRINCIPAL=random_forest \
  N_ESTIMATORS=200 \
  MAX_DEPTH=8 && \
make docker-run-inference
```

---

## Pruebas Unitarias

### Ejecutar todas las pruebas

```bash
# Todos los tests
uv run pytest -v

# Solo training
uv run pytest -v src/training/test

# Solo inference
uv run pytest -v src/inference/test

# Con cobertura
uv run pytest --cov=src src/
```

### Estructura de tests

Cada módulo tiene tests unitarios que prueban **funciones puras** con datos sintéticos:

#### `src/training/test/test_train.py` — 8 tests
- `test_preparar_datos_elimina_target_de_features` ✓
- `test_preparar_datos_conserva_features` ✓
- `test_preparar_datos_target_correcto` ✓
- `test_preparar_datos_lanza_error_sin_target` ✓
- `test_construir_modelo_random_forest_retorna_tipo_correcto` ✓
- `test_construir_modelo_linear_retorna_tipo_correcto` ✓
- `test_construir_modelo_random_forest_respeta_hiperparametros` ✓
- `test_construir_modelo_tipo_invalido_lanza_error` ✓

#### `src/inference/test/test_inference.py` — 5 tests
- `test_preparar_datos_elimina_columna_target` ✓
- `test_preparar_datos_sin_columna_target` ✓
- `test_generar_predicciones_con_modelo_mock` ✓
- `test_resumen_predicciones` ✓
- `test_guardar_predicciones` ✓

---

## Comandos de Desarrollo (Makefile)

### Linting

```bash
make lint              # Pylint en terminal
make lint-report       # Pylint con reporte en archivo
```

### Formateo de código

```bash
make format-black      # Formatea con Black
make format-black-check # Verifica sin modificar
make format-ruff       # Formatea con Ruff
make format-ruff-check # Verifica con Ruff
```

### Testing

```bash
make run-test          # Ejecuta pytest -v (todos los tests)
```

### Utilidades

```bash
make tree              # Muestra estructura del proyecto
make help              # Listado de todos los comandos disponibles
```

---

## Arquitectura de Código

### Principios de diseño

1. **Modularidad:** Cada paso del pipeline es un módulo independiente
2. **Testabilidad:** Funciones puras sin efectos secundarios
3. **Configurabilidad:** Argumentos CLI en lugar de valores hardcodeados
4. **Escalabilidad:** Containerización con Docker para reproducibilidad
5. **Calidad:** Linting, formateo y tests automáticos

### Flujo de datos

```
items.csv, categories.csv, shops.csv, sales_train.csv, test.csv
    ↓
[PREPROCESSING] → merge tables + validación + feature engineering
    ↓
parquet files
    ├── datos_entreno.parquet (80%)
    ├── datos_validacion.parquet (20%)
    └── datos_inferencia.parquet (test)
    ↓
[TRAINING] → preparar_datos() → construir_modelo() → evaluar()
    ↓
modelo_random_forest.joblib (se guarda el principal)
    ↓
[INFERENCE] → preparar_datos() → predecir() → resumen_predicciones()
    ↓
predicciones_batch.csv
```

---

## Dependencias Principales

| Librería | Versión | Uso |
|----------|---------|-----|
| pandas | ≥3.0.0 | Manipulación de datos |
| numpy | ≥2.4.1 | Operaciones numéricas |
| scikit-learn | ≥1.8.0 | Modelado y evaluación |
| joblib | (via sklearn) | Serialización de modelos |
| pyarrow | ≥23.0.0 | Archivos Parquet |
| pytest | (via pyproject.toml) | Tests unitarios |
| black | ≥26.1.0 | Formateo de código |
| pylint | ≥4.0.4 | Análisis de código |
| ruff | ≥0.15.0 | Formateo y linting |

---

## Calidad de Código

### Resultados de análisis estático

- **Pylint:** 10.00/10
- **Black:** Conforme
- **Ruff:** Conforme

### Cobertura de tests

```
src/training/test/ — 8/8 tests ✓ (100%)
src/inference/test/ — 5/5 tests ✓ (100%)
```

---

## Logs de Ejecución

Los logs de cada ejecución se guardan en `artifacts/logs/` con timestamps:

```bash
artifacts/logs/
├── training_20260228_120000.log
├── inference_20260228_120500.log
└── prep_20260228_120100.log
```

Cada log contiene:
- Timestamp de inicio/fin
- Métricas de desempeño (RMSE, mejora, etc.)
- Errores y advertencias
- Rutas de archivos generados

---

## Notas Importantes

### PYTHONPATH en Docker

El `PYTHONPATH=/app/src` en los Dockerfiles permite que Python encuentre los módulos:
- `training`, `inference`, `utils` directamente en imports
- Consistente con la configuración de pytest en `pyproject.toml`

### Argparse vs CLI directo

Todos los módulos usan `__main__.py` con argparse para CLI consistente:
```bash
python -m training --modelo random_forest --n-estimators 200
python -m inference --datos data/inference/nuevos.parquet
```

### Separación de I/O y lógica

Las funciones en `train.py` e `inference.py` son **puras** (sin I/O):
- `preparar_datos(df)` ← recibe DataFrame en memoria
- `construir_modelo(tipo)` ← retorna instancia de modelo
- `predecir(modelo, datos)` ← no accede a archivos

El I/O (read/write de archivos) ocurre en `main()`.

---

## Contacto y Contribuciones

Para sugerencias o reportes de bugs, crear un Issue o Pull Request en el repositorio.

---

**Última actualización:** Febrero 2026


