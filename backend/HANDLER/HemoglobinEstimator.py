import joblib
import numpy as np
import pandas as pd
import os
from ETL.input_preprocess import extract_features_from_image
class HemoglobinHandler:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'weights/best_randomforest_model.pkl')
        self.model = joblib.load(os.path.abspath(model_path))
        self.rmse = 1.97  # Set your model’s RMSE here

    def classify_severity(self, hgb):
        if hgb < 3: return "Inconclusive"
        if hgb < 8: return "Severe"
        elif hgb < 11: return "Moderate"
        elif hgb < 12: return "Mild"
        elif hgb < 20: return "Normal"
        else: return "Inconclusive"

    def predict_hgb(self, image_bytes):
        try:
            features = extract_features_from_image(image_bytes, debug=False)
            prediction = self.model.predict(features)[0]

            severity = self.classify_severity(prediction)
            if severity == "Inconclusive":
                return {
                "predicted_hemoglobin": "--",
                "anemia_severity": severity,
                "estimated_model_rmse": "--"
            }
            else:
                return {
                    "predicted_hemoglobin": round(prediction, 2),
                    "anemia_severity": severity,
                    "estimated_model_rmse": round(self.rmse, 2)
                }
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")