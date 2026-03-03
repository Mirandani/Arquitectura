# src/preprocessing/Dockerfile
FROM python:3.13.12

WORKDIR /app

ENV PYTHONPATH=/app/src

COPY . /app

# Instalar dependencias 
RUN pip install --no-cache-dir \
    joblib==1.5.3 \
    matplotlib==3.10.8 \
    numpy==2.4.2 \
    pandas==3.0.0 \
    scikit-learn==1.8.0 \
    scipy==1.17.0 \
    PyYAML==6.0.3 \
    seaborn==0.13.2 \
    pyarrow==11.0.0 \
# Nota: El script prep.py se montará desde el host con -v $(pwd):/app
# No se copia al contenedor para facilitar el desarrollo iterativo

# Definir punto de entrada fijo
ENTRYPOINT ["python", "src/preprocessing/prep.py"]

# Comando por defecto
CMD []
