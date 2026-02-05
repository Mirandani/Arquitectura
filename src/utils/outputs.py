import pandas as pd

# función para guardado de datasets parquet
def guardar_dataset(df, ruta):
    """Guarda el dataframe en parquet e imprime confirmación."""
    df.to_parquet(ruta, index=False)
    print(f"Guardado: {ruta}  Dimensiones: {df.shape}")