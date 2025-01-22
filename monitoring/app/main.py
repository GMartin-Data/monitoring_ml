from datetime import datetime, timezone
from enum import Enum
from logging import getLogger
from typing import Any

from fastapi import FastAPI, HTTPException, status
import pandas as pd
from pydantic import BaseModel, Field, validator

from model_utils import load_model, predict


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
    """,
    version="1.0.0",
)


# Enum classes for data validation and OpenAPI documentation
class Island(str, Enum):
    TORGERSEN = "Torgersen"
    BISCOE = "Biscoe"
    DREAM = "Dream"


class Sex(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class Species(str, Enum):
    ADELIE = "Adelie"
    CHINSTRAP = "Chinstrap"
    GENTOO = "Gentoo"


class FeaturesInput(BaseModel):
    """
    Input features for penguin species prediction.
    All measures must be positive numbers.
    """

    island: Island = Field(..., description="The island where the penguin was observed")
    bill_length_mm: float = Field(
        ...,
        gt=0,
        le=100,  # Determine it with sample data
        description="Bill length in millimeters",
    )
    bill_depth_mm: float = Field(
        ...,
        gt=0,
        le=50,  # Determine it with sample data
        description="Bill depth in millimeters",
    )
    flipper_length_mm: float = Field(
        ...,
        gt=0,
        le=500,  # Determine it with sample data
        description="Flipper length in millimeters",
    )
    body_mass_g: float = Field(
        ...,
        gt=0,
        le=10_000,  # Determine it with sample data
        description="Body mass in grams",
    )
    sex: Sex = Field(..., description="Sex of the penguin")

    # ⚠️ REWORK WHAT FOLLOWS TO Pydantic V2
    @validator("*")
    def check_nan_none(cls, v: Any) -> Any:
        """Validate that no field is NaN or None."""
        if pd.isna(v) or v is None:
            raise ValueError("Field cannot be NaN or None")
        return v

    class Config:
        schema_extra = {
            "example": {
                "island": "Biscoe",
                "bill_length_mm": 45.2,
                "bill_depth_mm": 15.5,
                "flipper_length_mm": 198,
                "body_mass_g": 4500,
                "sex": "Male",
            }
        }


class PredictionOutput(BaseModel):
    """
    Prediction output containing:
        - the predicted penguin species
        - an auto-generated timestamp
    """

    species: Species = Field(..., description="Predicted penguin species")
    timestamp: datetime = Field(
        default_factory=datetime.now(timezone.utc),
        description="Timestamp of the prediction",
    )

    class Config:
        schema_extra = {
            "example": {"species": "Gentoo", "timestamp": "2025-01-22T10:00:00.000Z"}
        }


# API Endpoints
## ⚠️ REWORK WHAT FOLLOWS TO Pydantic V2
## Startup event to load model
@app.on_event("startup")
async def startup_event():
    """Load the model when the API starts."""
    global model
    try:
        model = load_model()
        logger.info("✅ Model successfully loaded!")
    except Exception as e:
        logger.exception(f"❌ Failed to load model: {str(e)}")
        raise RuntimeError("⚠️ Failed to load model!")


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

    Args:
        features_input: input data

    Returns:
        predicted species and timestamp

    Raises:
        HTTPException if model prediction fails
    """
    try:
        # Convert input data to a pandas DataFrame
        X_input_dict = features_input.model_dump()
        X_input = pd.DataFrame(X_input_dict, index=[0])

        # Make prediction
        y_pred = predict(model, X_input)

        return PredictionOutput(species=y_pred, timestamp=datetime.now(timezone.utc))

    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error occured during prediction",
        )
