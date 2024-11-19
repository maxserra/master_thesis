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



if __name__ == "__main__":

    PROCESSED_DATA_FOLDER_PATH = Path("../data/processed")

    all_land_df = pd.read_parquet(PROCESSED_DATA_FOLDER_PATH.joinpath("CWatM_data", "all_land.parquet"))
    forcings_land_df = pd.read_parquet(PROCESSED_DATA_FOLDER_PATH.joinpath("CWatM_data", "forcings_land.parquet"))
    outputs_land_df = pd.read_parquet(PROCESSED_DATA_FOLDER_PATH.joinpath("CWatM_data", "outputs_land.parquet"))

    data_df = pd.concat((all_land_df, forcings_land_df, outputs_land_df), axis=1)

    INPUTS_COLUMNS = list(all_land_df.columns)
    FORCINGS_COLUMNS = list(forcings_land_df.columns)
    OUTPUTS_COLUMNS = list(outputs_land_df.columns)

    for inputs_columns_split in np.array_split(INPUTS_COLUMNS, len(INPUTS_COLUMNS)):

        for forcings_columns_split in np.array_split(FORCINGS_COLUMNS, 4):

            input_cols = inputs_columns_split.tolist() + forcings_columns_split.tolist()

            scores_df = compute_bivariate_scores(data_df,
                                                 input_cols=input_cols,
                                                 output_cols=OUTPUTS_COLUMNS,
                                                 dst_file_path=PROCESSED_DATA_FOLDER_PATH.joinpath("bivariate_metrics", "CWatM", "measures_global.csv"),
                                                 return_all=True)

    N_OF_SHUFFLES = 20

    for shuffled_id in range(N_OF_SHUFFLES):
        shuffled_data_df = utils.shuffle_data(data_df)

        for inputs_columns_split in np.array_split(INPUTS_COLUMNS, len(INPUTS_COLUMNS)):

            for forcings_columns_split in np.array_split(FORCINGS_COLUMNS, 4):

                input_cols = inputs_columns_split.tolist() + forcings_columns_split.tolist()

                shuffled_scores_df = compute_bivariate_scores(shuffled_data_df,
                                                            input_cols=input_cols,
                                                            output_cols=OUTPUTS_COLUMNS,
                                                            dst_file_path=PROCESSED_DATA_FOLDER_PATH.joinpath("bivariate_metrics",
                                                                                                                "CWatM",
                                                                                                                "shuffled",
                                                                                                                f"measures_global-{shuffled_id}.csv"),
                                                            return_all=False)
