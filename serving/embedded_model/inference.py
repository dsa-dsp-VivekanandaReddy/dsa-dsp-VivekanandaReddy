import joblib
import numpy as np
import pandas as pd


def predict(diabetes_dataframe: pd.DataFrame, model: str) -> np.ndarray:
    model = joblib.load('../../models/diabetes_rf.pkl')

    predictions = model.predict(diabetes_dataframe)
    return predictions
