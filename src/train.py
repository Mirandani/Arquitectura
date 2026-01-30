"""
src/train.py
    La entrada del script son datos data/prep. 
    La salida del script es un modelo entrenado. 
    Puedes checar el código de este blog Save and Load Machine Learning Models in Python with scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib


# Read data from data/prep/...
datos_entreno = pd.read_parquet('data/prep/datos_entreno.parquet')
datos_validacion = pd.read_parquet('data/prep/datos_validacion.parquet')
datos_prueba_final = pd.read_parquet('data/prep/datos_prueba_final.parquet')


X_entreno = datos_entreno.drop(['item_cnt_month'], axis=1)
Y_entreno = datos_entreno['item_cnt_month']

X_validacion = datos_validacion.drop(['item_cnt_month'], axis=1)
Y_validacion = datos_validacion['item_cnt_month']

X_prueba = datos_prueba_final.drop(['item_cnt_month'], axis=1)


# Entrenamiento del modelo
print("Entrenando el modelo...")
# REGRESIÓN LINEAL
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_entreno, Y_entreno)

# Evaluación del modelo
prediccion_lineal = modelo_lineal.predict(X_validacion)
error_lineal = np.sqrt(mean_squared_error(Y_validacion, prediccion_lineal))
print(f"RMSE Regresión Lineal: {error_lineal:.4f}")

# RANDOM FOREST
modelo_forest = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
modelo_forest.fit(X_entreno, Y_entreno)

prediccion_forest = modelo_forest.predict(X_validacion)
error_forest = np.sqrt(mean_squared_error(Y_validacion, prediccion_forest))
print(f"RMSE Random Forest: {error_forest:.4f}")

# save model joblib model.joblib
print("Guardando el modelo entrenado...")
joblib.dump(modelo_forest, 'models/modelo_random_forest.joblib')
print("Modelo guardado exitosamente en 'models/modelo_random_forest.joblib'")