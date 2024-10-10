from typing import List
import numpy as np
import pandas as pd
from functools import reduce


def shuffle_data(df_in: pd.DataFrame):

    df = df_in.copy()

    cols_to_shuffle = [col for col in df.columns if col not in ["lat", "lon"]]

    for col in cols_to_shuffle:
        
        values = df[col].values
        np.random.shuffle(values)
        df[col] = values

    return df


def get_metric_from_df_list(
    df_list: List[pd.DataFrame],
    metric: str
):
    return reduce(lambda x1, x2: pd.concat([x1[metric], x2[metric]], axis=1), df_list)
