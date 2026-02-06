# Makefile para tareas comunes de desarrollo

# Declaración de tareas como phony para evitar conflictos con archivos del mismo nombre
.PHONY: lint lint-report format format-check help

help:
	@echo "Comandos disponibles:"
	@echo "  make lint          - Ejecuta pylint y muestra resultados en terminal"
	@echo "  make lint-report   - Ejecuta pylint y guarda reporte en pylint_report.txt"
	@echo "  make format        - Formatea código con black"
	@echo "  make format-check  - Verifica formato sin modificar archivos"

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

