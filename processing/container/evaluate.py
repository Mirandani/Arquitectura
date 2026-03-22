import json
import os
import pathlib
import joblib
import tarfile
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

if __name__ == "__main__":
    
    # Rutas ScriptProcessor
    model_path = "/opt/ml/processing/model/model.tar.gz"
    test_path = "/opt/ml/processing/test/datos_inferencia.parquet"
    output_dir = "/opt/ml/processing/evaluation"
    
    # Extraer modelo
    print(f"Extrayendo modelo desde: {model_path}")
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    # Cargar el modelo extraido
    modelo = joblib.load("modelo_xgboost.joblib")
    
    # Cargar datos de prueba
    print(f"Cargando datos de prueba desde: {test_path}")
    df_test = pd.read_parquet(test_path)
    
    # Separar target
    columna_target = "item_cnt_month"
    X_test = df_test.drop([columna_target], axis=1)
    y_test = df_test[columna_target]
    
    # Predicciones y métricas
    print("Generando predicciones y calculando RMSE")
    predicciones = modelo.predict(X_test)
    
    # Cálculo rmse y std
    mse = mean_squared_error(y_test, predicciones)
    rmse = mse ** 0.5
    std = np.std(y_test - predicciones)
    print(f"RMSE calculado: {rmse:.4f}")
    
    # JSON de evaluación
    report_dict = {
        "regression_metrics": {
            "rmse": {
                "value": float(rmse), 
                "standard_deviation": float(std)
            },
        },
    }
    
    # Guardar archivo JSON
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = os.path.join(output_dir, "evaluation.json")
    
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
        
    print(f"Reporte de evaluación guardado en: {evaluation_path}")