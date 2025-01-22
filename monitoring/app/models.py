from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd
from sqlmodel import Field as SQLModelField, SQLModel
from pydantic import Field as PydanticField, validator


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


# BASE CLASS FOR ALL MODELS (to stick to DRY principle)
class FeaturesBase(SQLModel):
    """
    Base model containing all common penguin measurements and validations.
    This serves as our single source of truth for field definitions.
    """

    island: Island = PydanticField(
        ..., description="The island where the penguin was observed"
    )
    bill_length_mm: float = PydanticField(
        ...,
        gt=0,
        le=100,
        description="Bill length in millimeters - must be between 0 and 100",
    )
    bill_depth_mm: float = PydanticField(
        ...,
        gt=0,
        le=50,
        description="Bill depth in millimeters - must be between 0 and 50",
    )
    flipper_length_mm: float = PydanticField(
        ...,
        gt=0,
        le=500,
        description="Flipper length in millimeters - must be between 0 and 500",
    )
    body_mass_g: float = PydanticField(
        ...,
        gt=0,
        le=10000,
        description="Body mass in grams - must be between 0 and 10000",
    )
    sex: Sex = PydanticField(..., description="Sex of the penguin (Male/Female)")

    @validator("*")
    def check_nan_none(cls, v: Any) -> Any:
        """
        Validates that no field contains NaN or None values.
        The '*' means this validator applies to all fields.
        """
        if pd.isna(v) or v is None:
            raise ValueError("Fields cannot be NaN or None")
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


# API Input model
class FeaturesInput(FeaturesBase):
    """
    API input model that inherits all penguin measurements.
    No need to redefine the fields - they're all inherited from the base model.
    """

    pass


# Database model
class PredictionRecord(FeaturesBase, table=True):
    """Database model extending the base features."""

    id: int | None = SQLModelField(default=None, primary_key=True)
    predicted_species: Species = SQLModelField(
        ..., description="The species predicted by the model"
    )
    timestamp: datetime = SQLModelField(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the prediction was made",
    )


# Database Output model
class PredictionOutput(SQLModel):
    """
    Prediction output containing:
        - the predicted penguin species
        - the prediction's id in database
    """

    species: Species = PydanticField(..., description="Predicted penguin species")
    prediction_id: int = PydanticField(
        ..., description="Database ID of the saved prediction"
    )

    class Config:
        schema_extra = {"example": {"species": "Gentoo", "prediction_id": 1}}
