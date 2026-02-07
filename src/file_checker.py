# src/file_checker.py
"""Script para verificar la existencia y tamaño de archivos clave en el
proyecto."""

import os

# Lista de archivos a verificar según tus rutas
rutas = [
    "data/prep/datos_entreno.parquet",
    "data/prep/datos_validacion.parquet",
    "data/inference/datos_inferencia.parquet",
]

print("-" * 50)
print(f"{'ARCHIVO':<40} | {'TAMAÑO (MB)':<10}")
print("-" * 50)

for ruta in rutas:
    if os.path.exists(ruta):
        # Obtener tamaño en bytes y convertir a Megabytes
        bytes_size = os.path.getsize(ruta)
        mb_size = bytes_size / (1024 * 1024)
        print(f"{ruta:<40} | {mb_size:.2f} MB")
    else:
        print(f"{ruta:<40} | ❌ NO EXISTE")


print("-" * 50)
