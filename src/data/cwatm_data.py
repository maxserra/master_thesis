import pandas as pd


def process_inputs_df(
    df_in: pd.DataFrame
) -> pd.DataFrame:

    df = df_in.copy()

    # remove rows where Band1 is missing. Band1 represents land cover
    df = df[~df["Band1"].isna()]

    # remove all remaining rows with any missing value
    df = df.dropna(axis=0, how="any")

    df = df.drop(columns=["crs", "Band1"])

    return df

