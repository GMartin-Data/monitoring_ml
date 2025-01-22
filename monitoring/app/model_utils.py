import joblib
import pandas as pd


def load_model(model_path: str = "model.joblib") -> "model":
    """Utility function to load the model."""
    model = joblib.load(model_path)
    return model


def predict(model: "model", data: pd.DataFrame) -> str:
    """Generate model's prediction.

    Args:
        model: the trained model
        data: the data to predict on

    Returns:
        str: the penguin's specie.
    """
    y_pred = model.predict(data)
    return y_pred
