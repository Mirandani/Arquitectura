# --------------------------------------------------------------------------
# Base image
# Ubuntu 22.04 provee un entorno Linux robusto con Python 3.10
# --------------------------------------------------------------------------
FROM ubuntu:22.04

MAINTAINER Azucena <azucenaorduna@gmail.com>

# Evitar prompts interactivos durante la instalación de paquetes de Linux
ENV DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------
# Dependencias del sistema
# --------------------------------------------------------------------------
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
         build-essential \
         libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Crear aliases 'python' y 'pip' apuntando a las versiones Python 3.
RUN ln -s /usr/bin/python3 /usr/bin/python

# --------------------------------------------------------------------------
# Dependencias Python 
# --no-cache-dir evita guardar el cache de pip en la imagen, reduciendo su tamaño.
# Las versiones están fijadas para garantizar reproducibilidad.
# --------------------------------------------------------------------------
RUN pip3 --no-cache-dir install \
    numpy==1.26.4 \
    scipy==1.11.4 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    xgboost==2.0.3 \
    fastparquet==2023.10.1 \
    flask==3.0.0 \
    gunicorn==21.2.0 \
    joblib==1.3.2

# --------------------------------------------------------------------------
# Variables de entorno
# PATH: Agrega /opt/program al PATH para que SageMaker encuentre 'train' y 'serve'
# --------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# --------------------------------------------------------------------------
# Código de la aplicación
# Copia los scripts de entrenamiento e inferencia al directorio /opt/program.
# SageMaker invoca 'train' o 'serve' dependiendo del tipo de job.
# --------------------------------------------------------------------------
COPY src/training /opt/program/
COPY src/utils /opt/program/utils/

WORKDIR /opt/program

# Asegurar permisos de ejecución para los scripts de entrada
RUN chmod +x /opt/program/train
RUN chmod +x /opt/program/serve