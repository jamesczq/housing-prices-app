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

# Random state  
RANDOM_STATE = 123

# Test ratio of total data set
TEST_SIZE = 0.2

# Num of cross-validation folds
CV = 5