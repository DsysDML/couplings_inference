import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
import itertools

from torch import Tensor

## DCA and Contact Prediction Functions

def matrix_to_table(F2:np.ndarray) -> pd.DataFrame:
    """
    Convert a symmetric score matrix into a table format.

    Args:
        F2 (ndarray): Square score matrix. Only the upper triangular part (i < j) is used.

    Returns: 
        pd.DataFrame: Table with columns:
                      - 'i': row index
                      - 'j': column index
                      - 'F2': score value at (i, j)
    """
    Nv = F2.shape[0]
    data = []
    
    for i in range(Nv):
        for j in range(i + 1, Nv):
            data.append([i, j, F2[i, j]])
    
    return pd.DataFrame(data, columns=['i', 'j', 'F2'])


def custom_table(table: pd.DataFrame, 
                 ascending: bool =True, 
                 zero_index: bool =True) -> pd.DataFrame:
    """
    Create a custom table for scores and sequence distances.

    Args:
        table (pd.DataFrame): Input table with at least three columns: ['i', 'j', score_column].
        ascending (bool, default=True): If True, sort scores in ascending order. If False, sort in descending order.
        zero_index (bool, default=True): If True, indices in `i` and `j` are assumed to start from 0.
                                         If False, indices are assumed to start from 1 (and will be shifted to 0-based).

    Returns:
    pd.DataFrame: Custom table indexed by (i, j), with columns:
                  - 'rank': rank of the pair based on score ordering
                  - '|i-j|': absolute sequence distance
                  - score_column: score values
    """
    score_name = table.columns[-1]
    rank_name = 'rank'
    
    # ensure integer indices
    table['i'] = table['i'].astype(int)
    table['j'] = table['j'].astype(int)
    
    if not zero_index:
        table['i'] = table['i'] - 1
        table['j'] = table['j'] - 1
    
    # sort by score
    table = table.sort_values(by=score_name, ascending=ascending)
    
    # add sequence distance
    table.insert(2, '|i-j|', table['j'] - table['i'])
    
    # reset and reassign rank
    table = table.reset_index(drop=True).reset_index(drop=False)
    table.set_index(['i', 'j'], inplace=True)
    table.columns = [rank_name, '|i-j|', score_name]
    table[rank_name] = table[rank_name].astype(int)
    
    return table


def filter_seq_distance(table: pd.DataFrame, 
                        k: int =0) -> pd.DataFrame:
    """
    Filter a custom table by sequence distance.

    Args:
        table (pd.DataFrame): Custom table with column '|i-j|'.
        k (int, default=0): Minimum sequence distance threshold. Only pairs with |i-j| > k are kept.

    Returns:
        pd.DataFrame: Filtered table with reindexed ranks.
    """
    table_copy = table[table['|i-j|'] > k].copy()
    table_copy['rank'] = pd.factorize(table_copy['rank'])[0]  # reassign ranks
    return table_copy


def ROC_curve(xt: np.ndarray, 
              x: np.ndarray, 
              normalize=True) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the Receiver Operating Characteristic (ROC) curve and its area.

    Args:
        xt (ndarray): Ground-truth binary labels (0 = negative, nonzero = positive).
        x  (ndarray): Prediction scores.
        normalize (bool, default=True): If True, normalize false positives and true positives to [0, 1].

    Returns
        fp (ndarray): False positive counts (normalized if `normalize=True`).
        tp (ndarray): True positive counts (normalized if `normalize=True`).
        s (float): Area under the ROC curve.
    """
    N = xt.size
    if np.isnan(x).sum() != 0:
        return np.zeros(N + 1), np.zeros(N + 1), np.nan

    sorted_indices = np.argsort(x)[::-1]
    xt = xt[sorted_indices]
    tp, fp = np.zeros(N + 1), np.zeros(N + 1)
    s = 0.0

    for i in range(1, N + 1):
        if xt[i - 1] != 0:
            fp[i], tp[i] = fp[i - 1], tp[i - 1] + 1
        else:
            fp[i], tp[i] = fp[i - 1] + 1, tp[i - 1]
            s += tp[i - 1]

    s /= (tp[-1] * fp[-1])

    if normalize:
        fp /= np.max(fp)
        tp /= np.max(tp)

    return fp, tp, s

def PPV_curve(xt: np.ndarray, 
              x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the Positive Predictive Value (PPV) curve and its area.

    Args:
        xt (ndarray): Ground-truth binary labels.
        x  (ndarray): Prediction scores.

    Returns
        p (ndarray): Number of predictions considered (1 to N).
        ppv (ndarray): Positive predictive values for each rank.
        s (float): Area under the PPV curve.
    """
    N = xt.size
    if np.isnan(x).sum() != 0:
        return np.zeros(N + 1), np.zeros(N + 1), np.nan
    
    sorted_indices = np.argsort(x)[::-1]
    xt = xt[sorted_indices]

    ppv, p = np.zeros(N), np.zeros(N)
    s = 0.0

    for n in range(1, N+1):
        p[n-1] = n
        ppv[n-1] = xt[:n].sum() / n
        s += ppv[n-1]
    
    return p, ppv, s

# Gaussian Mixture Fitting

def hidden_col_sum(X: Tensor, 
                   W: Tensor, 
                   a: int) -> Tensor:
    """ Given random samples "X" evaluates the sum of weights on the column in "W" corresponding to a given hidden node "a"

    Args:
        X (Tensor): Random samples.
        W (Tensor): Weight matrix.
        a (int): Hidden node index.
    
    Returns:
        Sum of weights of "W" in the hidden column "a" evaluated in samples "X".
    """
    device = W.get_device()
    all_v = torch.arange(W.shape[1], device=device, dtype=torch.int32)
    X_samples = W[X, all_v, a]

    return X_samples.sum(axis=1).to('cpu').numpy() 

def fit_gaussian_mixture(W: Tensor, 
                         a: int, 
                         splits: int =0, 
                         sigma: float=5, 
                         nsteps: int=100) -> tuple[np.ndarray, np.ndarray]:
    """Fit a higher-order gaussian mixture to the distribution of randomly drawn weights from a hidden column.

    Args:
        W (Tensor): Weight matrix.
        a (int): Hidden node index.
        splits (int): Number of "bifurcations" to construct the gaussian mixture. The total number of gaussians will be q^{splits}.
    returns:
        Sum
    """
    q, Nv, Nh = W.shape
    W_a = W[:,:,a]

    std = np.linalg.norm(W_a)/np.sqrt(q)
    x = np.linspace(-std*sigma, std*sigma, nsteps)

    # returns simple gaussian, if no splits are asked
    if splits == 0:
        return x, norm.pdf(x, loc=0, scale=std)
    # creating a gaussian mixture if more than a split is required    
    F = np.linalg.norm(W_a, axis=0)
    ind_max = np.argsort(F)[-splits:][::-1]
    std_mod = np.sqrt(std**2 - (np.sum(F[ind_max]**2)/q) ) 
    combinations = itertools.product(*W_a[:, ind_max].T)
    mean_combinations = np.array([np.sum(combination) for combination in combinations])
    gauss_mod = np.zeros(nsteps)
    for mean in mean_combinations:
        gauss_mod += (1/q**splits)*norm.pdf(x, loc=mean, scale=std_mod)
    return x, gauss_mod





