import pandas as pd
import constants


def clean_df(df:pd.DataFrame, drop_dup=False):
    if drop_dup:
        return df.drop_duplicates(
            subset=constants.NUM_COLS + constants.CAT_COLS + [constants.Y_COL],
            inplace=True
            )
    return df


def get_x_y(df:pd.DataFrame):
    x = df[constants.NUM_COLS + constants.CAT_COLS]
    y = df[constants.Y_COL]
    return x, y


