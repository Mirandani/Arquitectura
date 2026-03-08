#!/usr/bin/env python
# Servidor de inferencia Flask para SageMaker BYOC para XGBoost.


from __future__ import print_function

import io
import os
import flask
import pandas as pd
import joblib

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

class ScoringService(object):
    """Implementa el patrón singleton para cargar el modelo una sola vez."""
    model = None

    @classmethod
    def get_model(cls):
        """Carga el modelo .joblib desde la ruta de SageMaker."""
        if cls.model is None:
            ruta_modelo = os.path.join(model_path, "modelo_xgboost.joblib")
            cls.model = joblib.load(ruta_modelo)
        return cls.model

    @classmethod
    def predict(cls, input_df):
        """Ejecuta la predicción sobre el DataFrame recibido."""
        clf = cls.get_model()
        return clf.predict(input_df)

# Aplicación Flask
app = flask.Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    """Health check: 200 si el modelo cargó, 404 si no."""
    health = ScoringService.get_model() is not None
    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def transformation():
    """Endpoint de inferencia para datos en formato CSV."""
    if flask.request.content_type == "text/csv":
        data_raw = flask.request.data.decode("utf-8")
        s = io.StringIO(data_raw)
        # Se lee sin header
        data_df = pd.read_csv(s, header=None)
    else:
        return flask.Response(
            response="This predictor only supports CSV data", 
            status=415, 
            mimetype="text/plain"
        )

    # Ejecutar la predicción.
    predictions = ScoringService.predict(data_df.values)

    # Convertir el array de predicciones a CSV y devolverlo como response.
    out = io.StringIO()
    pd.DataFrame({"results": predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")