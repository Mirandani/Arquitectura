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
MODELO_PRINCIPAL  ?= random_forest
N_ESTIMATORS      ?= 50
MAX_DEPTH         ?= 10
RANDOM_STATE      ?= 42
ENTRADA           ?= data/prep/datos_entreno.parquet
VALIDACION        ?= data/prep/datos_validacion.parquet
SALIDA_TRAIN      ?= artifacts/models/modelo_random_forest.joblib

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

docker-run-inference:
	# monta código, datos y artefactos para ejecutar inferencia
	docker run --rm \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/artifacts:/app/artifacts \
		inferencia:latest

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

