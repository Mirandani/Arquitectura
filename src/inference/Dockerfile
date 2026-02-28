# Dockerfile for inference module
# Uses CLI arguments

FROM python:3.13.12

# Establecer variables de entorno para evitar problemas comunes con Python en contenedores
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar uv en el contenedor
RUN pip install --no-cache-dir uv

# copiar archivos de configuración del proyecto
COPY pyproject.toml uv.lock ./

# Instalar dependencias del proyecto usando uv
RUN uv sync --frozen --no-install-project

# copiar el código fuente, sin datos ni artefactos
COPY ../../src/ ./src/

# instalar las dependencias del proyecto, incluyendo el paquete del proyecto
RUN uv sync --frozen

# arrancar siempre con uv; el comando por defecto lanza el script de entrenamiento
ENTRYPOINT ["uv", "run", "--"]
CMD ["python", "src/inference/inference.py"]
