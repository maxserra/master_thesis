from typing import List
import pandas as pd
import plotly.graph_objects as go


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
                      xaxis_title="dddd",
                      yaxis_title="aaa",
                      showlegend=False)

    fig.add_trace(go.Scatter(x=df[default_x],
                             y=df[default_y],
                             mode="markers"))

    buttons_row, buttons_col = [], []

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
            }
        ]
    )

    fig.show()
