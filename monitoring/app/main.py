from logging import getLogger

from fastapi import FastAPI, HTTPException, status
import pandas as pd

from .database import DatabaseManager
from .models import FeaturesInput, PredictionOutput
from .utils import load_model, predict


# Instanciate logger
logger = getLogger(__name__)


# Instanciate app
app = FastAPI(
    title="Penguin Species Predictor",
    description="""
    This API predicts penguin species based on:
        - physical measurement
        - location data
        - sex
    It uses a machine learning model trained on the Palmer Penguins dataset.
    🌐 Documentation link: https://github.com/allisonhorst/palmerpenguins
    The records are then stored in a SQLite database.
    """,
    version="1.0.0",
)


# Initialize database manager
db = DatabaseManager()


# API Endpoints
## ⚠️ REWORK WHAT FOLLOWS TO Pydantic V2
## Startup event to load model
@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    global model
    try:
        logger.info("⏳ Starting application initialization...")
        db.create_db_and_tables()
        model = load_model()
        logger.info("✅ Application startup completed successfully!")
    except Exception as e:
        logger.error(f"❌ Application startup failed: {str(e)}")
        raise RuntimeError("⚠️ Failed to initialize application!")


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Check API Health",
)
async def health_check() -> dict[str, str | bool]:
    """Check if the API is healthy and the model is loaded."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post(
    "/predict",
    response_model=PredictionOutput,
    responses={
        201: {"description": "Successful prediction", "model": PredictionOutput},
        422: {"description": "Validation error in input data"},
        500: {"description": "Internal servor error during prediction"},
    },
    tags=["Prediction"],
    summary="Predict penguin species from input data",
)
async def predict_species(features_input: FeaturesInput) -> PredictionOutput:
    """
    Predicts penguin species based on:
        - physical measurement
        - location data
        - sex
    Stores input data and prediction into a SQLite database, adding a timestamp.

    Args:
        features_input: input data

    Returns:
        PredictionOutput:
            - species (str): the predicted species
            - prediction_id (int): the ID of the database's record.

    Raises:
        HTTPException if model prediction fails
    """
    try:
        # Convert input data to a pandas DataFrame
        X_input_dict = features_input.model_dump()
        X_input = pd.DataFrame(X_input_dict, index=[0])

        # Make prediction
        y_pred = predict(model, X_input)

        # Save to database
        record = db.save_prediction(
            features_input=features_input.model_dump(), predicted_species=y_pred
        )

        return PredictionOutput(species=y_pred, prediction_id=record.id)

    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error occured during prediction",
        )
