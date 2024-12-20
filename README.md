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

To understand the overall folder strcuture, see the Project Organization section below.

To understand the work done within the scope of this thesis, go to `notebooks` and follow the numbering. The `.ipynb` files have quite descriptive names which indicate their purpose. These also include descriptive text and comments along the code, although not consistently. It should be possible to execute them sequentially as each of them builds upon the previous ones.

There are two know exepctions to this, which prevents the notebooks of being run to completion:
1. The data limitation mentioned above for `05-Lea-CWatM_data_processing.ipynb`
2. And `08-CWatM_mask_land_and_explore_data.ipynb`, where some scatterplot-generating functions expect measures to be available under `data/processed/bivariate_metrics/CWatM` and the file `data/processed/CWatM_data/chanleng_regions.csv` to exist. These are generated in notebooks `09-CWatM_compute_measures.ipynb` and `11-CWatM_explore_chanleng.ipynb`, respectively.

The notebooks up to and including `04` are the preliminary work done to test the methods. These are not part of the final thesis but are kept here for informative purposes.

In the notebooks from `05` to `11`, the core work can be found. From `05` to `08` the data is processed and prepared for processing and modelling. Notebooks `09` and `10`, together with the two scripts under `scripts` (numbered according to the respective notebook) are used to produce the main results of the thesis. This includes the computation of the measures of dependence, testing their statistical significance and the visualization of these results. In notebook `11` the chanleng pattern are explored, the "chanleng regions" are created, the multi-class Random Forest classifier is trained and evaluated and the permutation feature importance is calculated.

Notebook `20` contains some exploratory and preliminary work on divisive clustering which did not yield the expected results and has not been further pursued within the scope of this thesis.

### Project Organization

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
    ├── .gitignore
    ├── .python-version
    ├── LICENSE
    ├── pyproject.toml     <- Poetry project file.
    ├── README.md          <- The top-level README for developers using this project.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment,
    │                         generated with `poetry export --output requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

## How to run the code in this project

This code in this project is mainly writen with Python, except for some exploratory work done with R (as a suitable library for SSPCA was found in R).

To enable reporducibility of the analysis, the package management tool `poetry` has been used for dependency management. We refer to the official documentation: https://python-poetry.org/docs/basic-usage/. With this setup, it is possible to recreate the Python virtual environment with the packages and the specific version used during the thesis. The relevant files for this are `pyproject.toml` and `.python-version`.

If you do not wish to use `poetry`, a `requirements.txt` is also provided. Make sure to use the correspondiong Python version `3.9.13` and it should work.

This has been developed on a Mac, so if you have issues running this on another OS, write me an email.
