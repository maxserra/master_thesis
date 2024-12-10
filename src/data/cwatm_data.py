import numpy as np
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


def process_measures_df(
    df_in: pd.DataFrame
) -> pd.DataFrame:
    
    df = df_in.copy()

    df = df.dropna(axis="index",
                   subset="pearson")

    df = df[df["MIC"] != 0]

    df = df[["pearson", "spearman", "MIC", "MAS", "MEV"]]

    return df


def compute_ranks_df(
    df_in: pd.DataFrame
) -> pd.DataFrame:
    
    df = df_in.copy()
    
    df = df.abs()
    ranks_df = df.rank(axis=0,
                       method="min",
                       ascending=False).astype(int)

    ranks_df.columns = [f"{col}_rank" for col in df.columns]

    return ranks_df


def compute_p_values_complete(actual_df,
                              shuffled_data_path,
                              region: str,
                              n_shuffles: int = 25):

    shuffled_measures_dfs = []

    for i in range(n_shuffles):
        shuffled_measures_df = pd.read_csv(shuffled_data_path.joinpath(f"measures_{region}-{i}.csv"),
                                        index_col=["input", "output"])
        shuffled_measures_df = process_measures_df(shuffled_measures_df)
        shuffled_measures_dfs.append(shuffled_measures_df)

    p_values_measures_df = compute_p_values(actual_df=actual_df,
                                            shuffled_dfs=shuffled_measures_dfs)

    return p_values_measures_df


def compute_p_values(actual_df, shuffled_dfs):
    # Number of permutations
    N = len(shuffled_dfs)
    
    # Initialize an empty DataFrame to store p-values, with the same index as actual_df
    p_values = pd.DataFrame(index=actual_df.index, columns=actual_df.columns)
    
    # Loop through each row and column in actual_df
    for row_index in actual_df.index:
        for col in actual_df.columns:
            # Actual value for this row and column
            actual_value = actual_df.loc[row_index, col]
            
            # Count how many shuffled values are greater than or equal to the actual value
            count_ge = sum(df.loc[row_index, col] >= actual_value for df in shuffled_dfs)
            
            # Compute the p-value with the "+1" adjustment
            p_value = (count_ge + 1) / (N + 1)
            
            # Store the p-value
            p_values.loc[row_index, col] = p_value
    
    p_values.columns = [f"{col}_p-value" for col in actual_df.columns]

    p_values = p_values.astype(float)

    return p_values


def control_FDR_benjamini_hochberg(
    p_values_series_in: pd.Series,
    alpha: float = 0.05
) -> pd.Series:

    p_values_series = p_values_series_in.copy()
    
    # Total number of tests
    m = len(p_values_series)
    
    # Sort the p-values and keep track of their original indices
    p_values_sorted = p_values_series.sort_values()
    indices = p_values_sorted.index  # MultiIndex (row, column)
    
    # Assign ranks to the p-values
    ranks = np.arange(1, m + 1)
    
    # Calculate BH critical values
    bh_critical_values = (ranks / m) * alpha
    
    # Determine where p-values are less than or equal to critical values
    significant = p_values_sorted <= bh_critical_values
    
    # Find the largest p-value that is significant
    try:
        max_significant_rank = ranks[significant][-1]
        # Threshold p-value
        threshold_p_value = p_values_sorted.iloc[max_significant_rank - 1]
        
        # All p-values less than or equal to threshold are significant
        significant_p_values = p_values_series <= threshold_p_value
    except IndexError:
        # Threshold p-value
        threshold_p_value = 0.0

        # No p-values are significant
        significant_p_values = pd.Series(False, index=p_values_series.index)
    
    # Prepare data for plotting
    bh_data = {
        "data": pd.DataFrame({
            'p_value': p_values_sorted.values,
            'rank': ranks,
            'bh_critical_value': bh_critical_values
        }),
        "alpha": alpha,
        "threshold_p_value": threshold_p_value
    }

    return significant_p_values, bh_data


def compute_non_linearity(
    df_in: pd.DataFrame
) -> pd.DataFrame:
    
    df = df_in.copy()

    df["MIC - p^2"] = df["MIC"] - df["pearson"]**2

    return df
