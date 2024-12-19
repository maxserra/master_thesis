# MSc Thesis repository
# Using information theory to identify process controls in Global Water Models

This repository compiles all relevant code and data for my MSc thesis.

Here I have compiled all the code; functions, scripts and notebooks written throughout this thesis. 

If you are reading this without having access to the written part of the thesis itself, and you would like to read it, write me an email.

### Preliminary remarks on data availability

This work uses data from different sources. When looking to reporoduce some parts of the analysis you might need this data.

1. Data from the previous work by Gnann, can be found in https://zenodo.org/record/7714885 and should be stored under `data/raw/ISIMIP_2b_aggregated_variables`
2. Data specific to this thesis containing the inputs used in the ISIMIP3a run of the CWatM have been provided by Ting Tang and Yoshihide Wada (currently at KAUST). This data should be stored under `data/raw/CWatM_ISIMIP3_4Thorsten`. This raw data is, as of today, not included in this repository. However, this raw and granular data is only needed to run notebook `05-Lea-CWatM_data_processing.ipynb` and does not prevent the rest of the analysis to be run.
3. Raw data openly available are:
    - ISIMIP3a climatic forcings: This can be collected by executing the notebook `06-CWatM_input_data_processing.ipynb` and will be stored under `data/raw/CWatM_input`
    - CWatM outputs under study: This can be collected by executing the notebook `07-CWatM_output_processing.ipynb` and will be stored under `data/raw/CWatM_input`


## How to naviagte this project

## Project Organization

    ├── data
    │   ├── processed      <- The final, processed data used for analysis and modeling.
    │   └── raw            <- The original, immutable data.
    │
    ├── notebooks          <- Jupyter notebooks used for analysis. More on these below.
    │
    ├── reports            <- Generated visuals for reporting
    │   ├── figures        <- Plots and other graphical represenations
    │   └── tables         <- Tabular represenations
    │
    ├── scripts            <- Python scripts used for analysis. More on these below.
    │
    ├── src                <- Source code for the functions uses in this project.
    │   ├── __init__.py    <- Makes `src` a Python module.
    │   │
    │   ├── data           <- Functions to generate, process or transform data.
    │   │
    │   ├── dependence_measures <- Functions to compute dependence measures.
    │   │
    │   ├── models         <- Functions and classes defining self-coded models 
    │   │
    │   └── visualization  <- Functions to create exploratory and results oriented visualizations
    │
    ├── .env               <- File with secrets to be loaded as environment variables.
    ├── .gitignore
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── pyproject.toml     <- Poetry project file.
    ├── README.md          <- The top-level README for developers using this project.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment,
    │                         generated with `poetry export --output requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------
