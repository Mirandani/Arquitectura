# processing/container/Dockerfile
# Imagen basada en python:3.11-slim con scikit-learn, pandas y numpy instalados vía pip.
# No necesitas ENTRYPOINT — ScriptProcessor ejecuta el script directamente con python3.
FROM python:3.11-slim

ENV PYTHONHASHSEED 0
ENV PYTHONIOENCODING UTF-8

# Instalar scikit-learn, pandas y numpy
RUN pip install --no-cache-dir \
    numpy==2.4.2 \
    pandas==3.0.0 \
    scikit-learn==1.8.0

# por que no ponemos mas cosas en el Dockerfile?