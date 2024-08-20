from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
import sklearn
import sklearn.compose
import sklearn.pipeline

from pathlib import Path
import joblib
import json
import pandas as pd
import re


import preprocessing

# Define constants config for machine learning experiments
RANDOM_STATE = 123

TEST_SIZE = 0.2  # len(Test data) / len(total data)

CV = 5  # Num of cross-validation folds


def train_model(
    x_train: pd.DataFrame, y_train: pd.DataFrame
) -> sklearn.pipeline.Pipeline:
    """Build (1) a feature transformer and (2) a Random Forest regression model,
    given training data.
    * Notice that the returned object is composed of the above two.

    Args:
        x_train (pd.DataFrame): _description_
        y_train (pd.DataFrame): _description_

    Returns:
        sklearn.pipeline.Pipeline: a pipeline of feature transformer and ML model
    """

    feature_transformer = preprocessing.define_feature_transformer()

    ml_pipeline = Pipeline(
        [
            ("Preprocessing", feature_transformer),
            ("Regression Model Random Forest", RandomForestRegressor(n_estimators=30)),
        ]
    )

    model = ml_pipeline.fit(x_train, y_train)

    return model


def eval_model(
    model: sklearn.pipeline.Pipeline, x: pd.DataFrame, y: pd.DataFrame
) -> str:
    """Evaluate model performance by estimation using cross-validation.

    Args:
        model (sklearn.pipeline.Pipeline): _description_
        x (pd.DataFrame): _description_
        y (pd.DataFrame): _description_

    Returns:
        str: a formatted string describing model performance
    """
    scores = cross_val_score(model, x, y, scoring="neg_mean_absolute_error", cv=CV)
    mae = -scores
    mu, sigma = mae.mean(), mae.std()

    str_out = f"Mean Absolute Error (MAE) estimated from cross-validation: "
    str_out += f"{mu:.2f} +/- {sigma:.2f}"
    print(str_out)

    return str_out


def main():
    """Perform the main actions of training, evaluating and saving ML model.
    * No return values; purely causing side-effects.
    """
    # 1. Training model
    # 1.1. Prepare training data X and y
    data_path = Path.cwd() / "data" / "housing.csv"

    df = pd.read_csv(data_path)
    df = preprocessing.clean_df(df)

    x, y = preprocessing.get_x_y(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # 1.2.
    model = train_model(x_train, y_train)

    # 2. Evaluating model
    str_cv_mae = eval_model(model, x, y)

    test_mae = mean_absolute_error(y_true=y_test, y_pred=model.predict(x_test))
    print(f"Test Mean Absolute Error (MAE): {test_mae:.2f}")

    # 3. Saving model to file
    # 3.1.
    model_path = Path.cwd() / "models" / "model.pkl"
    joblib.dump(model, model_path)
    print(f"Model (Random Forest Regressor) saved to: {model_path}")

    # 3.2. In addition, also saving model info, e.g., model components/performance
    model_info_path = Path.cwd() / "models" / "model_info.json"
    model_info = {
        "performance": str_cv_mae,
        "model-info": re.sub(" +", " ", str(model)).strip(),
    }
    with open(model_info_path, "w") as f:
        json.dump(model_info, f)
    print(f"Model info saved to: {model_info_path}")


if __name__ == "__main__":
    main()
