from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go

import src.data.utils as utils


def plot_scatter_with_dropdown(df: pd.DataFrame,
                               default_x: str,
                               default_y: str,
                               valid_x: List[str] = None,
                               valid_y: List[str] = None,
                               layout_height: int = 500,
                               layout_width: int = 800):

    # Check and prepare variables
    if (default_x not in df.columns) or (default_y not in df.columns):
        raise ValueError(f"Invalid input parameters, make sure that '{default_x}' and '{default_y}' are columns of df ({df.columns})")
    
    valid_x = list(df.columns.copy()) if valid_x is None else valid_x
    valid_y = list(df.columns.copy()) if valid_y is None else valid_y

    if not (set(valid_x).issubset(df.columns)) or not (set(valid_y).issubset(df.columns)):
        raise ValueError(f"Invalid input parameters, make sure that '{valid_x}' and '{valid_y}' are valid subsets of the columns of df ({df.columns})")

    # Plot
    fig = go.Figure()

    fig.update_layout(title="title here",
                      height=layout_height,
                      width=layout_width,
                      xaxis_title=default_x,
                      yaxis_title=default_y,
                      showlegend=False)

    fig.add_trace(go.Scatter(x=df[default_x],
                             y=df[default_y],
                             mode="markers",
                             marker={
                                "colorscale": "Viridis",
                                "opacity": 0.2,
                                "line": {"color": "black", "width": 0.25},
                                "color": df[default_x],
                                "colorbar": {"title": default_x}
                             }))

    buttons_row, buttons_col, buttons_color = [], [], []

    for row in valid_x:

        buttons_row.append({
            "label": row,
            "method": "update",
            "args": [{"x": [df[row]],
                      "type": "scatter"},
                     {"xaxis": {"title": row}}],
        })

    for col in valid_y:
        
        buttons_col.append({
            "label": col,
            "method": "update",
            "args": [{"y": [df[col]],
                      "type": "scatter"},
                     {"yaxis": {"title": col}}],
        })

    for col in valid_x + valid_y:

        buttons_color.append({
            "label": col,
            "method": "update",
            "args": [{"marker": {"colorscale": "Viridis",
                                 "opacity": 0.25,
                                 "color": df[col],
                                 "colorbar": {"title": col}}}]
        })

    fig.update_layout(
        updatemenus=[
            {
                "active": False,
                "buttons": buttons_row,
                "direction": "down",
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.35,
                "xanchor": "center",
                "y": 1.2,
                "yanchor": "top"
            },
            {
                "active": False,
                "buttons": buttons_col,
                "direction": "down",
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.65,
                "xanchor": "center",
                "y": 1.2,
                "yanchor": "top"
            },
            {
                "active": False,
                "buttons": buttons_color,
                "direction": "down",
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.85,
                "xanchor": "center",
                "y": 1.2,
                "yanchor": "top"
            }
        ],
        annotations=[
            {
                "text": "X variable",
                "x": 0.35,
                "xref": "paper",
                "xanchor": "center",
                "y": 1.25,
                "yref": "paper",
                "align": "left",
                "showarrow": False
            },
            {
                "text": "Y variable",
                "x": 0.65,
                "xref": "paper",
                "y": 1.25,
                "yref": "paper",
                "align": "left",
                "showarrow": False
            },
            {
                "text": "color",
                "x": 0.85,
                "xref": "paper",
                "y": 1.25,
                "yref": "paper",
                "align": "left",
                "showarrow": False
            }
        ]
    )

    fig.show()


def plot_measure_values_and_rank(measures_df: pd.DataFrame,
                                 measures: List[str],
                                 sort_values_by: str,
                                 n_top: int = 10):

    df = measures_df.abs()
    df = df[measures].sort_values(sort_values_by, ascending=False)
    df = df.head(n_top)

    df_rank = df.rank(axis=0, ascending=False)

    fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    axis[0].plot(df.T,
                 marker="o")
    axis[0].set_title("Measures value")
    axis[0].set_xlabel("Measure of dependence")
    axis[0].set_ylabel("Value")

    axis[1].plot(df_rank.T,
                 label=list(df_rank.T.columns),
                 marker="o")
    axis[1].set_title("Measures rank")
    axis[1].set_xlabel("Measure of dependence")
    axis[1].set_ylabel("Rank")
    axis[1].yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=n_top-2))
    axis[1].invert_yaxis() # Invert the y-axis to have rank 1 at the top

    # Move the legend to the right of the plot
    axis[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Adjust the layout to make room for the legend
    fig.tight_layout(rect=[0, 0, 1, 1])

    return fig


def plot_metric_values_and_rank_with_shuffle(scores_df: pd.DataFrame,
                                             metrics: List[str],
                                             shuffle_scores_df_list: List[pd.DataFrame],
                                             sort_values_by: str):

    scores_df = scores_df[metrics].sort_values(sort_values_by, ascending=False)
    scores_df_rank = scores_df.rank(axis=0)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    for i, col in enumerate(scores_df.T):
        axis[0].plot(metrics,
                     scores_df.T[col],
                     marker="o",
                     color=colors[i]
                     )
    axis[0].set_title("Metrics value")
    axis[0].set_xlabel("Metric")
    axis[0].set_ylabel("Value")

    for j in range(len(shuffle_scores_df_list)):

        shuffle_scores_df = shuffle_scores_df_list[j][metrics]

        for i, col in enumerate(shuffle_scores_df.T):
            axis[0].plot(metrics,
                         shuffle_scores_df.T[col],
                         alpha=0.2,
                         color=colors[i] # this is wrong, ordering is not matched to scores_df! TODO
                         )

    axis[1].plot(scores_df_rank.T,
                 label=list(scores_df_rank.T.columns),
                 marker="o")
    axis[1].set_title("Metrics rank")
    axis[1].set_xlabel("Metric")
    axis[1].set_ylabel("Rank")
    axis[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axis[1].legend(loc="lower right")

    fig.tight_layout()
    return fig


def plot_metric_baseline_and_value(
    scores_df: pd.DataFrame,
    metrics: List[str],
    shuffle_scores_df_list: List[pd.DataFrame],
    sort_values_by: str
):

    fig, axis = plt.subplots(nrows=1, ncols=len(metrics),
                             sharey=True,
                             figsize=(14, 4))

    for i, metric in enumerate(metrics):

        metric_value = scores_df.sort_values(sort_values_by, ascending=True)[metric]
        metric_baseline_values = utils.get_metric_from_df_list(shuffle_scores_df_list, metric)

        metric_all = pd.concat([metric_value.rename("value", inplace=True), metric_baseline_values], axis=1)

        import numpy as np

        axis[i].set_title(f"{metric}")
        axis[i].boxplot(metric_all.T,
                        vert=False,
                        labels=[str(index) for index in metric_all.index],
                        )
        axis[i].scatter(x=metric_value.values,
                        y=np.arange(len(metric_value)) + 1,
                        marker="X", c="red", s=100)
        axis[i].grid(axis="x")

        # for j, index in enumerate(metric_value.index):

        #     axis[j, i].set_title(f"{metric} for {index}")



    fig.tight_layout()
    return fig
