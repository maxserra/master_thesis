from typing import Dict
import pandas as pd


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
