"""This file defines Data Model: feature names, target name, 
expected input features: allowed values and data types."""

from enum import Enum
from pydantic import BaseModel
from typing import Optional

# Names of numeric feature columns
NUM_COLS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

# Names categorical feature columns
CAT_COLS = ["ocean_proximity"]

# Names of target column
Y_COL = "median_house_value"


# Defined the allowed values for input feature ocean_proximity
class OceanProximity(Enum):
    VALUE1 = "<1H OCEAN"
    VALUE2 = "INLAND"
    VALUE3 = "NEAR OCEAN"
    VALUE4 = "NEAR BAY"
    VALUE5 = "ISLAND"


# Define the allowed data-types for input features
class HouseProperty(BaseModel):
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    housing_median_age: Optional[float] = None
    total_rooms: Optional[float] = None
    total_bedrooms: Optional[float] = None
    population: Optional[float] = None
    households: Optional[float] = None
    median_income: Optional[float] = None
    ocean_proximity: Optional[OceanProximity] = None
