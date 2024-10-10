from typing import List
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
from IPython.display import display

from src.dependence_measures import bivariate


def compute_scores(args):
    """Compute bivariate scores for a single pair of input and output."""
    df, input_col, output_col = args

    # pearson
    pearson_score = bivariate.pearson(df[input_col], df[output_col])

    # spearman
    spearman_score = bivariate.spearman(df[input_col], df[output_col])

    # maximal correlation
    # mc_score = bivariate.maximal_correlation_SVD(df[input_col], df[output_col])

    # mutual information
    mi_score = bivariate.mutual_information_sklearn(df[input_col].values.reshape(-1, 1),
                                                    df[output_col].values)

    # maximal information coefficient
    mic_scores = bivariate.maximal_information_coefficient(df[input_col], df[output_col])

    return {
        "input": input_col,
        "output": output_col,
        "pearson": pearson_score,
        "spearman": spearman_score,
        # "maximal correlation (SVD)": mc_score,
        "mutual information (sklearn)": mi_score,
        "MIC": mic_scores["MIC"],
        "MAS": mic_scores["MAS"],
        "MEV": mic_scores["MEV"],
        "MCN_general": mic_scores["MCN_general"]
    }

def compute_bivariate_scores(df: pd.DataFrame,
                             input_cols: List[str],
                             output_cols: List[str]) -> pd.DataFrame:
    """Compute bivariate scores for input-output combinations

    Args:
        df (pd.DataFrame): _description_
        input_cols (List[str]): _description_
        output_cols (List[str]): _description_

    Returns:
        pd.DataFrame: The different scores for the different combinations of inputs-outputs
    """
    # Generate all combinations of input-output columns
    combinations = list(product(input_cols, output_cols))
    args_list = [(df, input_col, output_col) for input_col, output_col in combinations]

    # Use ProcessPoolExecutor to parallelize the computation
    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(tqdm(executor.map(compute_scores, args_list), 
                            total=len(combinations), desc="Computing input-output combinations"))

    # Convert the results into a DataFrame
    records_df = pd.DataFrame.from_records(results)
    records_df.set_index(["input", "output"], inplace=True)

    return records_df


def compute_bivariate_scores_on_file_generator(path_rglob):
    scores_list = []

    for file in sorted(path_rglob):

        data_df = pd.read_csv(file, index_col=False)

        scores_df = compute_bivariate_scores(data_df, ["0"], ["1"])
        scores_df["file"] = file.name
        scores_df.set_index("file", append=False, drop=True, inplace=True)

        scores_list.append(scores_df)

    scores_df = pd.concat(scores_list)
    scores_df = scores_df.sort_index()

    scores_df_styled = scores_df.style.format(lambda x: f"{x:.2f}")#.background_gradient(cmap="OrRd", axis=0))

    display(scores_df_styled)
    return scores_df_styled, scores_df
