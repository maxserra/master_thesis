from pathlib import Path
from typing import List, Dict
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


def preprocess_data(df: pd.DataFrame,
                    columns_renamer: Dict[str, str] = None) -> pd.DataFrame:

    out_df = df.copy()

    if "netrad" in out_df.columns:
        out_df["netrad"] = out_df["netrad"] * 12.87
        
    if "evap" in out_df.columns:
        out_df = out_df[out_df["evap"] < 10000]
    
    if "qr" in out_df.columns:
        out_df.loc[out_df["qr"] < 0, "qr"] = 0

    if columns_renamer is not None:
        out_df.rename(columns=columns_renamer, inplace=True)

    return out_df
