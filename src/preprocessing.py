import pandas as pd

from schema import CAT_COLS, NUM_COLS, Y_COL

def clean_df(df:pd.DataFrame, drop_dup=False):
    if drop_dup:
        return df.drop_duplicates(
            subset=NUM_COLS + CAT_COLS + [Y_COL],
            inplace=True
            )
    return df


def get_x_y(df:pd.DataFrame):
    x = df[NUM_COLS + CAT_COLS]
    y = df[Y_COL]
    return x, y


