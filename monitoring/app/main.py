from typing import Literal

from fastapi import FastAPI
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
