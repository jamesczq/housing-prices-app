from enum import Enum
from pydantic import BaseModel
from typing import Optional

# Define numeric feature columns, categorical feature columns, target columns
NUM_COLS = [
    'longitude', 
    'latitude', 
    'housing_median_age', 
    'total_rooms', 
    'total_bedrooms', 
    'population', 
    'households', 
    'median_income']

CAT_COLS = ['ocean_proximity']

Y_COL = 'median_house_value'


class OceanProximity(Enum):
    VALUE1 = "<1H OCEAN"
    VALUE2 = "INLAND"
    VALUE3 = "NEAR OCEAN"
    VALUE4 = "NEAR BAY"
    VALUE5 = "ISLAND"


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