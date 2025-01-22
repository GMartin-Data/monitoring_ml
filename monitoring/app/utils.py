from logging import getLogger
from pathlib import Path

import joblib
import pandas as pd


logger = getLogger(__name__)


def load_model():
    """
    Load the trained model.

    Uses the current file's location to determine the correct path to model.joblib,
    ensuring it works regardless of where the application is started from
    """
    # Construct the absolute path to 'model.joblib'
    current_dir = Path(__file__).parent.absolute()
    model_path = current_dir / "model.joblib"

    try:
        logger.info(f"⏳ Attempting to load model from {model_path}")
        model = joblib.load(model_path)
        logger.info("✅ Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise


def predict(model: "model", data: pd.DataFrame) -> str:
    """Generate model's prediction.

    Args:
        model: the trained model
        data: the data to predict on

    Returns:
        str: the penguin's specie.
    """
    label_mapping = {
        0: "Adelie",
        1: "Chinstrap",
        2: "Gentoo"
    }
    y_pred = model.predict(data)
    return label_mapping[y_pred[0]]
