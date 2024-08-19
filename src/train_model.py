
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pathlib import Path

import joblib
import json
import pandas as pd
import re 

from schema import CAT_COLS, NUM_COLS
import preprocessing
import constants


def define_feature_transformer():
    num_pipeline = Pipeline([
        ("Imputer", SimpleImputer(strategy="median")),
        ("Std_Scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('One-hot', OneHotEncoder(handle_unknown='ignore'))
    ])

    feature_transformer = ColumnTransformer([
        ('Numeric Feature Transform', num_pipeline, NUM_COLS),
        ('Categorical Feature Transform', cat_pipeline, CAT_COLS)
    ])
    
    return feature_transformer


def train_model(x_train, y_train):

    feature_transformer = define_feature_transformer()

    ml_pipeline = Pipeline([
        ('Preprocessing', feature_transformer),
        ('Regression Model Random Forest', RandomForestRegressor(n_estimators=30))
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
    str_out = f"Mean Absolute Error (MAE) estimated from cross-validation: "
    str_out += f"{mu:.2f} +/- {sigma:.2f}"
    print(str_out)
    return str_out


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

    str_cv_mae = eval_model(model, x, y)

    test_mae = mean_absolute_error(y_true=y_test, y_pred=model.predict(x_test))
    print(f"Test Mean Absolute Error (MAE): {test_mae:.2f}")

    model_path = Path.cwd()/"models"/"model.pkl"
    joblib.dump(model, model_path)
    print(f"Model (Random Forest Regressor) saved to: {model_path}")

    model_info_path = Path.cwd()/"models"/"model_info.json"
    model_info = {
        'performance': str_cv_mae,
        'model-info': re.sub(' +', ' ', str(model)).strip()
    }
    with open(model_info_path, "w") as f:
        json.dump(model_info, f)
    print(f"Model info saved to: {model_info_path}")


if __name__ == "__main__":
    main()



