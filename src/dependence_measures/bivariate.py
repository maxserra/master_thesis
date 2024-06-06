from typing import Dict
import math
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from minepy import MINE


def pearson(x, y):
    return stats.pearsonr(x, y).statistic


def spearman(x, y):
    return stats.spearmanr(x, y).statistic


def _maxCorr(data, X, Y):  # NOT USED
    # taken from https://github.com/tokamaster/maximal-correlation/blob/main/correlation/maxcorr.py
    
    # inputs = data with x in col 1 and y in col 2, alphabets X and Y
    lx = len(X)
    ly = len(Y)
    # x indexes rows and y indexes columns
    Pxy = [[0 for _ in range(ly)] for _ in range(lx)]
    n = len(data)
    for i in range(1, n+1):
        indx = [ix for ix, x in enumerate(X) if x == data[i-1][0]]
        indy = [iy for iy, y in enumerate(Y) if y == data[i-1][1]]
        Pxy[indx[0]][indy[0]] += 1
    Pxy = [[cell/sum(row) for cell in row]
           for row in Pxy]  # empirical joint distribution of data
    Px = [sum(row) for row in Pxy]  # empirical mariginal distribution of data
    # empirical mariginal distribution of data
    Py = [sum(col) for col in zip(*Pxy)]
    B = [[Pxy[r][s]*(1/math.sqrt(Px[r]))*(1/math.sqrt(Py[s]))
          for s in range(ly)] for r in range(lx)]
    for r in range(lx):
        for s in range(ly):
            if math.isnan(B[r][s]) or math.isinf(B[r][s]):
                B[r][s] = 0  # change all NaNs or infinities to 0
    U, S, V = np.linalg.svd(B)
    return S[1]  # output = maximal correlation


def maximal_correlation_SVD(x, y,
                            n_bins: int = 10):

    Pxy = np.histogram2d(x, y, bins=n_bins)[0]
    Pxy = Pxy / Pxy.sum()
    Px = Pxy.sum(axis=0)
    Py = Pxy.sum(axis=1)

    B = Pxy * ((1 / np.sqrt(Px)) * (1 / np.sqrt(Py)))

    B = np.nan_to_num(B, nan=0, posinf=0, neginf=0)

    U, S, V = np.linalg.svd(B)

    return S[1]


def mutual_information_sklearn(x, y):
    return mutual_info_regression(x, y)[0]


def maximal_information_coefficient(x, y) -> Dict[str, float]:
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(x, y)

    return {"MIC": mine.mic(),
            "MAS": mine.mas(),
            "MEV": mine.mev(),
            "MCN_general": mine.mcn_general(),
            "TIC": mine.tic(norm=True)}
