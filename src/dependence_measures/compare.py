from typing import List
from itertools import product

import pandas as pd

from src.dependence_measures import bivariate


def compute_bivariate_scores(df: pd.DataFrame,
                              input_cols: List[str],
                              output_cols: List[str]) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        input_cols (List[str]): _description_
        output_cols (List[str]): _description_

    Returns:
        pd.DataFrame: The different scores for the different combinations of inputs-outputs
    """

    records = []
    combinations = product(input_cols, output_cols)

    for input_col, output_col in combinations:

        # pearson
        pearson_score = bivariate.pearson(df[input_col], df[output_col])

        # spearman
        spearman_score = bivariate.spearman(df[input_col], df[output_col])

        # maximal correlation
        mc_score = bivariate.maximal_correlation_SVD(df[input_col], df[output_col])

        # mutual information
        mi_score = bivariate.mutual_information_sklearn(df[input_col].values.reshape(-1, 1),
                                                        df[output_col].values)

        # maximal information coefficent
        mic_score = bivariate.maximal_information_coefficient(df[input_col], df[output_col])


        records.append({"input": input_col,
                        "output": output_col,
                        "pearson": pearson_score,
                        "spearman": spearman_score,
                        "maximal correlation (SVD)": mc_score,
                        "mutual information (sklearn)": mi_score,
                        "maximal information coefficient": mic_score})
    
    records_df = pd.DataFrame.from_records(records)
    records_df.set_index(["input", "output"], inplace=True)

    return records_df