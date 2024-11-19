import sys
sys.path.append("..")

from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import dataframe_image as dfi
import matplotlib.pyplot as plt


import src.data.utils as utils
from src.dependence_measures.compare import compute_bivariate_scores


def compute_shuffled_scores_region(data_df,
                                   regions_df,
                                   region,
                                   n_shuffles: int):

    region_indices = regions_df[regions_df["region"] == region].index
    region_indices = set(region_indices).intersection(data_df.index)

    region_data_df = data_df.loc[list(region_indices)]

    for shuffled_id in range(n_shuffles):
        shuffled_region_data_df = utils.shuffle_data(region_data_df)

        for inputs_columns_split in np.array_split(INPUTS_COLUMNS, 25):

            for forcings_columns_split in np.array_split(FORCINGS_COLUMNS, 2):

                input_cols = inputs_columns_split.tolist() + forcings_columns_split.tolist()

                shuffled_scores_df = compute_bivariate_scores(shuffled_region_data_df,
                                                             input_cols=input_cols,
                                                             output_cols=OUTPUTS_COLUMNS,
                                                             dst_file_path=PROCESSED_DATA_FOLDER_PATH.joinpath("bivariate_metrics",
                                                                                                               "CWatM",
                                                                                                               "shuffled",
                                                                                                               f"measures_{region}-{shuffled_id}.csv"),
                                                             return_all=False)


if __name__ == "__main__":

    PROCESSED_DATA_FOLDER_PATH = Path("../data/processed")

    all_land_df = pd.read_parquet(PROCESSED_DATA_FOLDER_PATH.joinpath("CWatM_data", "all_land.parquet"))
    forcings_land_df = pd.read_parquet(PROCESSED_DATA_FOLDER_PATH.joinpath("CWatM_data", "forcings_land.parquet"))
    outputs_land_df = pd.read_parquet(PROCESSED_DATA_FOLDER_PATH.joinpath("CWatM_data", "outputs_land.parquet"))

    data_df = pd.concat((all_land_df, forcings_land_df, outputs_land_df), axis=1)

    INPUTS_COLUMNS = list(all_land_df.columns)
    FORCINGS_COLUMNS = list(forcings_land_df.columns)
    OUTPUTS_COLUMNS = list(outputs_land_df.columns)

    RAW_DATA_FOLDER_PATH = Path("../data/raw")

    domains_df = pd.read_csv(RAW_DATA_FOLDER_PATH.joinpath("ISIMIP_2b_aggregated_variables", "domains.csv"))
    domains_df = domains_df[["lon", "lat", "domain_days_below_1_0.08_aridity_netrad"]]
    regions_df = domains_df.rename(columns={"domain_days_below_1_0.08_aridity_netrad": "region"})
    regions_df = regions_df.set_index(["lon", "lat"])

    regions = regions_df["region"].unique()

    for region in regions:

        region_indices = regions_df[regions_df["region"] == region].index
        region_indices = set(region_indices).intersection(data_df.index)

        print(region, len(region_indices))

        region_data_df = data_df.loc[list(region_indices)]

        for inputs_columns_split in np.array_split(INPUTS_COLUMNS, 25):

            for forcings_columns_split in np.array_split(FORCINGS_COLUMNS, 2):

                input_cols = inputs_columns_split.tolist() + forcings_columns_split.tolist()

                scores_df = compute_bivariate_scores(region_data_df,
                                                     input_cols=input_cols,
                                                     output_cols=OUTPUTS_COLUMNS,
                                                     dst_file_path=PROCESSED_DATA_FOLDER_PATH.joinpath("bivariate_metrics",
                                                                                                       "CWatM",
                                                                                                       f"measures_{region}.csv"),
                                                     return_all=True)

    N_OF_SHUFFLES = 20

    REGION = "wet warm"
    compute_shuffled_scores_region(data_df=data_df,
                                    regions_df=regions_df,
                                    region=REGION,
                                    n_shuffles=N_OF_SHUFFLES)
    REGION = "dry warm"
    compute_shuffled_scores_region(data_df=data_df,
                                    regions_df=regions_df,
                                    region=REGION,
                                    n_shuffles=N_OF_SHUFFLES)
    REGION = "dry cold"
    compute_shuffled_scores_region(data_df=data_df,
                                    regions_df=regions_df,
                                    region=REGION,
                                    n_shuffles=N_OF_SHUFFLES)
    REGION = "wet cold"
    compute_shuffled_scores_region(data_df=data_df,
                                    regions_df=regions_df,
                                    region=REGION,
                                    n_shuffles=N_OF_SHUFFLES)
