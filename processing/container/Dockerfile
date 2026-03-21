# processing/container/Dockerfile
# Dockerfile de tarea 06
# Imagen basada en python:3.11-slim con scikit-learn, pandas y numpy instalados vía pip.
# No necesitas ENTRYPOINT — ScriptProcessor ejecuta el script directamente con python3.
FROM python:3.11-slim

ENV PYTHONHASHSEED 0
ENV PYTHONIOENCODING UTF-8

# Instalar dependencias scipy, scikit-learn, pandas, numpy y joblib
RUN pip install --no-cache-dir \
    numpy==2.4.2 \
    pandas==3.0.0 \
    pyarrow>=23.0.0 \
    scikit-learn==1.8.0 \
    scipy==1.11.4 \
    joblib==1.3.2