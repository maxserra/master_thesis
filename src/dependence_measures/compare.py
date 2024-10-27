import os
import logging
from typing import List
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
from IPython.display import display

from src.dependence_measures import bivariate
from src.dependence_measures import mutual_information


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]-[%(name)s.%(funcName)s:%(lineno)d] - %(levelname)s - %(message)s") 


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
    mi_score = mutual_information.mutual_information_sklearn(df[input_col].values.reshape(-1, 1),
                                                             df[output_col].values)
    
    # normalized
    nmi_score = mutual_information.normalized_mutual_information(df[input_col], df[output_col])

    # maximal information coefficient
    mic_scores = bivariate.maximal_information_coefficient(df[input_col], df[output_col])

    return {
        "input": input_col,
        "output": output_col,
        "pearson": pearson_score,
        "spearman": spearman_score,
        # "maximal correlation (SVD)": mc_score,
        "mutual information (sklearn)": mi_score,
        "normalized mutual information": nmi_score,
        "MIC": mic_scores["MIC"],
        "MAS": mic_scores["MAS"],
        "MEV": mic_scores["MEV"],
        "MCN_general": mic_scores["MCN_general"]
    }


def compute_bivariate_scores(df: pd.DataFrame,
                             input_cols: List[str],
                             output_cols: List[str],
                             dst_file_path: str,
                             return_all: bool) -> pd.DataFrame:
    """Compute bivariate scores for input-output combinations, with caching.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        input_cols (List[str]): List of input column names.
        output_cols (List[str]): List of output column names.
        dst_file_path (str): Path to the file where results are stored.
        return_all (bool): If True, return all results; if False, return only the specified combinations.

    Returns:
        pd.DataFrame: DataFrame containing the scores for the combinations.
    """
    # Load existing DataFrame from dst_file_path if it exists
    if os.path.exists(dst_file_path):
        existing_df = pd.read_csv(dst_file_path, index_col=["input", "output"])
        
        logging.info(f"Loaded data found in {dst_file_path}")
    else:
        logging.info(f"No previous data found, starting fresh...")
        # Create an empty DataFrame with MultiIndex for consistency
        existing_df = pd.DataFrame()
        existing_df.index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['input', 'output'])

    # Generate all combinations of input-output columns
    combinations = list(product(input_cols, output_cols))
    combinations = set(combinations)

    # Get existing combinations from existing_df
    if not existing_df.empty:
        existing_combinations = set(existing_df.index.tolist())
    else:
        existing_combinations = set()

    # Compute new combinations by taking the difference
    new_combinations = combinations - existing_combinations

    logging.info(f"New combinations: '{new_combinations}', existing ones: '{existing_combinations}'")

    # Compute scores for new combinations
    if new_combinations:
        # Extract unique column names using set comprehension
        cols_in_new_combinations = {col for pair in new_combinations for col in pair}
        print(cols_in_new_combinations)
        # Select only the relevant columns from the original DataFrame
        relevant_df = df[list(cols_in_new_combinations)]
        # Prepare arguments for compute_scores
        args_list = [(relevant_df, input_col, output_col) for input_col, output_col in new_combinations]
        # Use ProcessPoolExecutor to parallelize the computation
        with ProcessPoolExecutor(max_workers=3) as executor:
            results = list(tqdm(executor.map(compute_scores, args_list),
                                total=len(new_combinations),
                                desc="Computing new input-output combinations"))

        # Convert the results into a DataFrame
        new_records_df = pd.DataFrame.from_records(results)
        new_records_df.set_index(["input", "output"], inplace=True)

        # Concatenate new results with existing DataFrame
        updated_df = pd.concat([existing_df, new_records_df])
    else:
        # No new combinations to compute
        updated_df = existing_df

    logging.info(f"Storing all data to {dst_file_path}")
    # Store updated_df to dst_file_path
    updated_df.to_csv(dst_file_path)

    # Return the desired DataFrame
    if return_all:
        logging.info(f"Returning all exisitng combinations")
        return updated_df
    else:
        logging.info(f"Returning only requested combinations")
        # Filter updated_df to include only the specified combinations
        mask = updated_df.index.isin(combinations)
        result_df = updated_df[mask]
        return result_df


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
