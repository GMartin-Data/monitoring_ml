from typing import Literal

from fastapi import FastAPI, status
import pandas as pd
from pydantic import BaseModel, Field

from model_utils import load_model, predict


app = FastAPI()


SPECIES = Literal["Adelie", "Chinstrap", "Gentoo"]
ISLAND = Literal["Torgersen", "Biscoe", "Dream"]
SEX = Literal["Male", "Female"]


class FeaturesInput(BaseModel):
    island: ISLAND
    bill_length_mm: float = Field(..., gt=0)
    bill_depth_mm: float = Field(..., gt=0)
    flipper_length_mm: float = Field(..., gt=0)
    body_mass_g: float = Field(..., gt=0)
    sex: SEX


class PredictionOutput(BaseModel):
    species: SPECIES
    

model = load_model()


@app.post(
    "/predict",
    response_model=PredictionOutput,
    responses={
        201: {"description": "prediction processed"}
    })
def give_prediction(features_input: FeaturesInput):
    X_input_dict = features_input.model_dump()
    X_input = pd.DataFrame(X_input_dict, index=[0])
    y_pred = predict(model, X_input)
    
    return PredictionOutput(y_pred)
