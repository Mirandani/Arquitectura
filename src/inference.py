"""
La entrada de este script son datos data/inference y el modelo entrenado model.joblib.
La salida de este script son predicciones en batch que se guardan en data/predictions.
"""

import pandas as pd
import numpy as np  
import joblib

# Cargar datos de inferencia
datos_inferencia = pd.read_parquet('data/inference/datos_inferencia.parquet')
datos_inferencia = datos_inferencia.drop(['item_cnt_month'], axis=1)

# Cargar el modelo entrenado
modelo_cargado = joblib.load('artifacts/models/modelo_random_forest.joblib')
print("Modelo cargado exitosamente.")

# Realizar predicciones batch
print("Realizando predicciones...")
predicciones = modelo_cargado.predict(datos_inferencia) 
# Agregar predicciones al DataFrame
datos_inferencia['item_cnt_month_pred'] = predicciones

# Guardar las predicciones en data/predictions
datos_inferencia[['ID', 'item_cnt_month_pred']].to_csv('data/predictions/predicciones_batch.csv', index=False)
print("Predicciones guardadas en 'data/predictions/predicciones_batch.csv'.")   

print(datos_inferencia.head())

#metricas de resumen
print("Resumen de predicciones:")
print(f"Predicciones - media: {np.mean(predicciones):.2f}, mediana: {np.median(predicciones):.2f}, min: {np.min(predicciones):.2f}, max: {np.max(predicciones):.2f}")