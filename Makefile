.PHONY: lint lint-report help

help:
	@echo "Comandos disponibles:"
	@echo "  make lint         - Ejecuta pylint y muestra resultados en terminal"
	@echo "  make lint-report  - Ejecuta pylint y guarda reporte en pylint_report.txt"

lint:
	uv run pylint --output-format=text src/ || true

lint-report:
	uv run pylint --output-format=text src/ > pylint_report.txt || true
	@echo "Reporte guardado en pylint_report.txt"
