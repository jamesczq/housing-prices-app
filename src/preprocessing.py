from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn

from typing import Tuple
import pandas as pd

from schema import CAT_COLS, NUM_COLS, Y_COL


def clean_df(df: pd.DataFrame, drop_dup=False) -> pd.DataFrame:
    """Clean a Pandas DataFrame.
    * For simplicity, we only do de-duplication here.
    * You can define other cleaning actions.

    Args:
        df (pd.DataFrame): _description_
        drop_dup (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    if drop_dup:
        return df.drop_duplicates(subset=NUM_COLS + CAT_COLS + [Y_COL], inplace=True)
    return df


def get_x_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separate features (X), target variables (y), given a dataframe.

    Args:
        df (pd.DataFrame): _description_

    Returns:
        Tuple[pd.DataFrame.pd.DataFrame]: _description_
    """
    x = df[NUM_COLS + CAT_COLS]
    y = df[Y_COL]
    return x, y


def define_feature_transformer() -> sklearn.compose.ColumnTransformer:
    """Define the actions to tranform raw features, numeric and categorical.

    Returns:
        sklearn.compose.ColumnTransformer: _description_
    """
    num_pipeline = Pipeline(
        [
            ("Imputer", SimpleImputer(strategy="median")),
            ("Std_Scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline([("One-hot", OneHotEncoder(handle_unknown="ignore"))])

    feature_transformer = ColumnTransformer(
        [
            ("Numeric Feature Transform", num_pipeline, NUM_COLS),
            ("Categorical Feature Transform", cat_pipeline, CAT_COLS),
        ]
    )

    return feature_transformer
