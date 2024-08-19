
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import preprocessing

import constants



def define_feature_transformer():
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    feature_transformer = ColumnTransformer([
        ('num', num_pipeline, constants.NUM_COLS),
        ('cat', cat_pipeline, constants.CAT_COLS)
    ])
    
    return feature_transformer


def train_model(x_train, y_train):

    feature_transformer = define_feature_transformer()

    ml_pipeline = Pipeline([
        ('preprocessing', feature_transformer),
        ('randomforest', RandomForestRegressor(n_estimators=30))
    ])

    model = ml_pipeline.fit(x_train, y_train)

    return model


def eval_model(model, x, y):
    scores = cross_val_score(
        model, 
        x, 
        y, 
        scoring='neg_mean_absolute_error',
        cv=constants.CV)
    mae = -scores
    mu, sigma = mae.mean(), mae.std()
    str_out = f"Training Mean Absolute Error (MAE): {mu:.2f} +/- {sigma:.2f}"
    print(str_out)


def main():
    # Train +  Eval + Save model
    data_path = Path.cwd()/"data"/"housing.csv"

    df = pd.read_csv(data_path)
    df = preprocessing.clean_df(df)

    x, y = preprocessing.get_x_y(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, 
        y, 
        test_size=constants.TEST_SIZE,
        random_state=constants.RANDOM_STATE
    )
    
    model = train_model(x_train, y_train)

    eval_model(model, x, y)

    test_mae = mean_absolute_error(y_true=y_test, y_pred=model.predict(x_test))
    print(f"Test Mean Absolute Error (MAE): {test_mae:.2f}")

    model_path = Path.cwd()/"models"/"model.pkl"
    joblib.dump(model, model_path)
    print(f"Model (Random Forest Regressor) -> {model_path}")

if __name__ == "__main__":
    main()



