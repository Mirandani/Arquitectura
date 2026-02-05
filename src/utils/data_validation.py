
import pandas as pd

# función para validación de datos nulos y vista previa
def validar_datos(df, nombre="DataFrame"):
    """Reporte de dimensiones, nulos y vista previa."""
    print(f"\n--- {nombre} {df.shape} ---")
    
    # Conteo de nulos
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    
    print(f"Nulos:\n{nulos if not nulos.empty else 'Ninguno'}")
    
    # Vista previa
    print("\nHead (3):")
    print(df.head(3))