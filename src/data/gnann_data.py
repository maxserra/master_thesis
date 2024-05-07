from pathlib import Path
from typing import List
from warnings import warn

import pandas as pd


def load_and_merge_geo_csv_to_df(data_path: Path,
                                 files: List[str],
                                 cols_to_keep: List[str] = None
                                 ) -> pd.DataFrame:

    out_df = None

    for file in files:
        temp_df = pd.read_csv(data_path.joinpath(f"{file}.csv"))
        
        if out_df is None:
            out_df = temp_df.copy()
        else:
            out_df = out_df.merge(right=temp_df, how="inner", on=["lat", "lon"])
    
    if cols_to_keep is not None:
        try:
            out_df = out_df[["lat", "lon"] + cols_to_keep]
        except KeyError as e:
            warn(e)
            return None

    return out_df
