# Dockerfile del modulo de entrenamiento de modelo de machine learning.  
# Usa argumentos CLI

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
# --frozen asegura que se instalen exactamente las versiones especificadas en uv.lock
# --no-install-project evita que se instale el paquete del proyecto, lo que es útil para evitar problemas de dependencias circulares durante el desarrollo
RUN uv sync --frozen --no-install-project

# copiar el código fuente, sin datos ni artefactos
COPY ../../src/ ./src/

# instalar las dependencias del proyecto, incluyendo el paquete del proyecto
RUN uv sync --frozen

# arrancar siempre con uv; -m training ejecuta src/training/__main__.py (con argparse)
# los argumentos de `docker run` se añaden al final del ENTRYPOINT
ENTRYPOINT ["uv", "run", "--", "python", "-m", "training"]
