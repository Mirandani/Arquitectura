# Makefile para tareas comunes de desarrollo

# Declaración de tareas como phony para evitar conflictos con archivos del mismo nombre
.PHONY: lint lint-report format format-check tree help

help:
	@echo "Comandos disponibles:"
	@echo "  make lint          - Ejecuta pylint y muestra resultados en terminal"
	@echo "  make lint-report   - Ejecuta pylint y guarda reporte en pylint_report.txt"
	@echo "  make format        - Formatea código con black"
	@echo "  make format-check  - Verifica formato sin modificar archivos"
	@echo "  make tree          - Muestra estructura del proyecto"

tree:
	@command -v tree >/dev/null 2>&1 && \
		tree -I '__pycache__|*.pyc|.git|data|artifacts|.pytest_cache|.venv|venv' -L 3 || \
		find . -not -path '*/\.*' -not -path '*/data/*' -not -path '*/__pycache__/*' -not -path '*/artifacts/*' | sort

# Contruir imagen de Docker para entrenamiento
docker-build-train:
	docker build -f src/training/Dockerfile -t entrenamiento:latest .

# Variables con valores por defecto; se pueden sobreescribir en la línea de comandos:
#   make docker-run-train MODELO_BASE=random_forest MODELO_PRINCIPAL=linear
#   make docker-run-train N_ESTIMATORS=200 MAX_DEPTH=6
MODELO_BASE       ?= linear
MODELO_PRINCIPAL  ?= xgboost
N_ESTIMATORS      ?= 100
MAX_DEPTH         ?= 6
RANDOM_STATE      ?= 42
ENTRADA           ?= data/prep/datos_entreno.parquet
VALIDACION        ?= data/prep/datos_validacion.parquet
SALIDA_TRAIN      ?= artifacts/models/modelo_xgboost.joblib

docker-run-train:
	# monta código, datos y artefactos para ejecutar entrenamiento
	docker run --rm \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/artifacts:/app/artifacts \
		entrenamiento:latest \
		--entrada $(ENTRADA) \
		--validacion $(VALIDACION) \
		--salida $(SALIDA_TRAIN) \
		--modelo-baseline $(MODELO_BASE) \
		--modelo-principal $(MODELO_PRINCIPAL) \
		--n-estimators $(N_ESTIMATORS) \
		--max-depth $(MAX_DEPTH) \
		--random-state $(RANDOM_STATE)

docker-build-inference:
	docker build -f src/inference/Dockerfile -t inferencia:latest .

# Variables con valores por defecto; se pueden sobreescribir en la línea de comandos:
#   make docker-run-inference DATOS=data/inference/otro.parquet
MODELO ?= artifacts/models/modelo_random_forest.joblib
DATOS  ?= data/inference/datos_inferencia.parquet
SALIDA ?= data/predictions/predicciones_batch.csv

docker-run-inference:
	# monta código, datos y artefactos para ejecutar inferencia
	docker run --rm \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/artifacts:/app/artifacts \
		inferencia:latest \
		--modelo $(MODELO) \
		--datos $(DATOS) \
		--salida $(SALIDA)

.PHONY: run-test
run-test:
	@echo "Ejecutando pruebas con pytest..."	
	uv run pytest -v

docker-build-inference:
	docker build -f src/inference/Dockerfile -t inferencia:latest .

# Variables con valores por defecto; se pueden sobreescribir en la línea de comandos:
#   make docker-run-inference DATOS=data/inference/otro.parquet
MODELO ?= artifacts/models/modelo_random_forest.joblib
DATOS  ?= data/inference/datos_inferencia.parquet
SALIDA ?= data/predictions/predicciones_batch.csv

docker-run-inference:
	# monta código, datos y artefactos para ejecutar inferencia
	docker run --rm \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/artifacts:/app/artifacts \
		inferencia:latest \
		--modelo $(MODELO) \
		--datos $(DATOS) \
		--salida $(SALIDA)

.PHONY: run-test
run-test:
	@echo "Ejecutando pruebas con pytest..."	
	uv run pytest -v

# Tarea para ejecutar pylint y mostrar resultados en terminal
lint:
	uv run pylint --output-format=text src/ || true

# Tarea para ejecutar pylint y guardar resultados en un archivo de texto
lint-report:
	uv run pylint --output-format=text src/ > pylint_report.txt || true
	@echo "Reporte guardado en pylint_report.txt"

# Tarea para formatear código con black
format-black:
	uv run black src/

# Tarea para verificar formato con black sin modificar archivos
format-black-check:
	uv run black --check --diff src/

# Lista solo archivos que necesitan formato (sin diff)
format-black-list:
	uv run black --check src/ || true

# Tarea para formatear con ruff
format-ruff:
	uv run ruff format src/

# Tarea para verificar formato con ruff sin modificar archivos
format-ruff-check:
	uv run ruff check --output-format=full src/ || true

# Contruir imagen de Docker para preprocesamiento
docker-build-prep:
	docker build -f src/preprocessing/Dockerfile -t preprocesamiento:latest .

# Variables con valores por defecto para preprocesamiento:
ITEMS_PATH      ?= data/raw/items_en.csv
CATEGORIES_PATH ?= data/raw/item_categories_en.csv
SHOPS_PATH      ?= data/raw/shops_en.csv
TRAIN_PATH      ?= data/raw/sales_train.csv
TEST_PATH       ?= data/raw/test.csv
OUT_TRAIN       ?= data/prep/datos_entreno.parquet
OUT_VAL         ?= data/prep/datos_validacion.parquet
OUT_INFER       ?= data/inference/datos_inferencia.parquet


docker-run-prep:
	# monta código y datos para ejecutar preprocesamiento
	docker run --rm \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/data:/app/data \
		preprocesamiento:latest \
		--items $(ITEMS_PATH) \
		--categories $(CATEGORIES_PATH) \
		--shops $(SHOPS_PATH) \
		--train $(TRAIN_PATH) \
		--test $(TEST_PATH) \
		--out-train $(OUT_TRAIN) \
		--out-val $(OUT_VAL) \
		--out-infer $(OUT_INFER)