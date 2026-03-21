#!/usr/bin/env python
# processing/container/evaluate.py

"""
Script de evaluación para SageMaker Processing Job.
Lee el modelo entrenado y los datos de test, calcula RMSE y lo guarda en JSON.
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rutas por defecto de SageMaker
DEFAULT_INPUT_PATH = "/opt/ml/processing/input"
DEFAULT_OUTPUT_PATH = "/opt/ml/processing/output"


# ========================================================================
# FUNCIÓN PRINCIPAL
# ========================================================================

def main(input_path, output_path):
    """
    Evalúa el modelo entrenado en el conjunto de test.
    
    Args:
        input_path: Ruta base de entrada (e.g., /opt/ml/processing/input)
        output_path: Ruta base de salida (e.g., /opt/ml/processing/output)
    """
    
    logger.info("=" * 70)
    logger.info("Iniciando evaluación del modelo")
    logger.info("=" * 70)
    logger.info(f"Input path:  {input_path}")
    logger.info(f"Output path: {output_path}")
    
    # Rutas de entrada
    model_dir = os.path.join(input_path, "model")
    test_dir = os.path.join(input_path, "test")
    
    # Rutas de salida
    output_eval_dir = os.path.join(output_path, "evaluation")
    eval_json_path = os.path.join(output_eval_dir, "evaluation.json")
    
    # Crear directorio de salida
    os.makedirs(output_eval_dir, exist_ok=True)
    
    try:
        # ====================================================================
        # 1. CARGAR MODELO
        # ====================================================================
        logger.info("\n[1/3] Cargando modelo entrenado...")
        
        model_path = os.path.join(model_dir, "modelo_xgboost.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        modelo = joblib.load(model_path)
        logger.info(f"✓ Modelo cargado: {model_path}")
        
        # ====================================================================
        # 2. CARGAR DATOS DE TEST
        # ====================================================================
        logger.info("\n[2/3] Cargando datos de test...")
        
        test_path = os.path.join(test_dir, "datos_inferencia.parquet")
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Datos de test no encontrados en: {test_path}")
        
        datos_test = pd.read_parquet(test_path)
        logger.info(f"✓ Datos de test cargados: {test_path}")
        logger.info(f"  - Forma: {datos_test.shape}")
        logger.info(f"  - Columnas: {list(datos_test.columns)}")
        
        # ====================================================================
        # 3. CALCULAR MÉTRICAS
        # ====================================================================
        logger.info("\n[3/3] Calculando métricas de evaluación...")
        
        # Columna target (variable a predecir)
        target_column = "item_cnt_month"
        
        if target_column not in datos_test.columns:
            raise ValueError(f"Columna target '{target_column}' no encontrada en datos de test")
        
        # Separar features y target
        X_test = datos_test.drop([target_column], axis=1)
        y_test = datos_test[target_column]
        
        logger.info(f"  - Features: {X_test.shape[1]} columnas")
        logger.info(f"  - Target: {target_column} con {len(y_test)} muestras")
        
        # Hacer predicciones
        y_pred = modelo.predict(X_test)
        
        # Calcular RMSE (Root Mean Squared Error)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        logger.info(f"\n✓ Métricas calculadas:")
        logger.info(f"  - MSE:  {mse:.4f}")
        logger.info(f"  - RMSE: {rmse:.4f}")
        
        # ====================================================================
        # 4. GUARDAR RESULTADOS EN JSON
        # ====================================================================
        logger.info(f"\nGuardando resultados en: {eval_json_path}")
        
        # Formato estándar de SageMaker para evaluation.json
        evaluation_result = {
            "regression_metrics": {
                "rmse": {
                    "value": float(rmse),
                    "standard_deviation": float(np.std(np.abs(y_test - y_pred)))
                },
                "mse": {
                    "value": float(mse)
                },
                "mae": {
                    "value": float(np.mean(np.abs(y_test - y_pred)))
                }
            }
        }
        
        # Guardar JSON
        with open(eval_json_path, 'w') as f:
            json.dump(evaluation_result, f, indent=4)
        
        logger.info(f"✓ Resultados guardados en JSON")
        logger.info(f"\n" + "=" * 70)
        logger.info("✓ EVALUACIÓN COMPLETADA EXITOSAMENTE")
        logger.info("=" * 70)
        logger.info(f"\nRMSE final: {rmse:.4f}")
        logger.info(f"Archivo de evaluación: {eval_json_path}")
        
        return rmse
        
    except Exception as e:
        logger.error("=" * 70)
        logger.error("✗ ERROR CRÍTICO EN LA EVALUACIÓN")
        logger.error("=" * 70)
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación de modelo para SageMaker Processing Job"
    )
    
    parser.add_argument(
        "--input-path",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Ruta de entrada (/opt/ml/processing/input)"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Ruta de salida (/opt/ml/processing/output)"
    )
    
    args = parser.parse_args()
    
    main(args.input_path, args.output_path)
