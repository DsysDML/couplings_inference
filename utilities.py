import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
import itertools
from typing import Tuple
from torch import Tensor
import h5py

# reading and loading the models
def get_saved_updates(model_fname: str) -> np.ndarray:
    """
    Retrieve the saved updates from a model file.

    Args:
        model_fname (str): Path to the model file.

    Returns:
        np.ndarray: Array of saved updates.
    """
    with h5py.File(model_fname, 'r') as f:
        updates = np.array([int(k.split('_')[1]) for k in f.keys()])
    return np.sort(updates)

def get_model_params(model_fname: str, update: int, device: str = 'cpu', dtype: torch.dtype =torch.float32) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Load the model parameters from a model file for a specific update.
    Args:
        model_fname (str): Path to the model file.
        update (int): Specific update to load.
        device (str): Device to load the parameters onto.
        dtype (torch.dtype): Data type of the parameters.
    Returns:
        Tuple[Tensor, Tensor, Tensor]: Weight matrix, visible bias, hidden bias.
    """
    with h5py.File(model_fname, 'r') as f:
        grp = f[f'update_{update}']
        weight_matrix = torch.tensor(grp['weight_matrix'][:], device=device, dtype=dtype)
        vbias = torch.tensor(grp['vbias'][:], device=device, dtype=dtype)
        hbias = torch.tensor(grp['hbias'][:], device=device, dtype=dtype)
    return weight_matrix, vbias, hbias

# RBM Fixing Gauge
def fix_gauge_RBM(W: Tensor, 
                  b: Tensor, 
                  c: Tensor, 
                  gauge: str ='zero-sum') -> None:
    """
    a Function to fix the gauge in Potts-Bernoulli RBMs

    Args:
        W (Tensor): weight matrix (dim: q x Nv x Nh).
        b (Tensor): visible bias (dim: q x Nv).
        c (Tensor): hidden bias (dim: Nh).
        gauge (str): name of the gauge that is going to be fixed. 
            It could be either 'zero-sum' or 'lattice-gas'.
    
    Returns:
        None
    """

    if gauge == 'zero-sum':
        A = W.mean(axis=0)
        bt = b.mean(axis=0)

    elif gauge == 'lattice-gas':
        # the zero is set in the last color
        A = W[-1,:,:]
        bt= b[-1,:]
    else:
        return 'Gauge does not exist'

    b -= bt
    c += A.sum(axis=0)
    W -= A

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
              normalize: bool =True) -> Tuple[np.ndarray, np.ndarray, float]:
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
              x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
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

# creating a ROC curve dependent on threshold of physical distance
def ROC_curve_distance(distance_table: pd.DataFrame, 
                       score_table: pd.DataFrame, 
                       k: int, 
                       threshold: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns the Receiver Operating Characteristic for contact prediction given a distance table and a threshold.

    Args:
        distance_table (pd.DataFrame): pairwise distance table in the custom fomrat.
        score_table (pd.DataFrame): score table in the custom format.
        k (int): threshold in the site distance.
        threshold (float): threshold in the physical distance in Armstrong units to consider a contact.
    
    Returns:
        fp (ndarray): False positive counts (normalized if `normalize=True`).
        tp (ndarray): True positive counts (normalized if `normalize=True`).
        s (float): Area under the ROC curve.
    """

    name = score_table.columns[-1]
    contact_table = distance_table[['|i-j|']].join(distance_table[['r']] < threshold)
    contact_table = contact_table.join(score_table[name])
    contact_table = contact_table[contact_table['|i-j|'] > k]
    xt, x = contact_table['r'].values, contact_table[name].values
    
    fp, tp, s = ROC_curve(xt, x)
    return fp, tp, s

# creating a ROC curve dependent on threshold of physical distance
def PPV_curve_distance(distance_table, score_table, k, threshold):
    """
    Returns the Receiver Operating Characteristic for contact prediction given a distance table and a threshold.

    Args:
        distance_table (pd.DataFrame): pairwise distance table in the custom fomrat.
        score_table (pd.DataFrame): score table in the custom format.
        k (int): threshold in the site distance.
        threshold (float): threshold in the physical distance in Armstrong units to consider a contact.

    Returns
        p (ndarray): Number of predictions considered (1 to N).
        ppv (ndarray): Positive predictive values for each rank.
        s (float): Area under the PPV curve.
    """

    name = score_table.columns[-1]
    contact_table = distance_table[['|i-j|']].join(distance_table[['r']] < threshold)
    contact_table = contact_table.join(score_table[name])
    contact_table = contact_table[contact_table['|i-j|'] > k]
    xt, x = contact_table['r'].values, contact_table[name].values
    
    p, ppv, s = PPV_curve(xt, x)
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
                         nsteps: int=100) -> Tuple[np.ndarray, np.ndarray]:
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

# Beyond the Gaussian Approximation Functions
def find_outliers_index(W: np.ndarray, 
                        z_score_threshold: float, 
                        sum_vis_dim: bool =True) -> list:
    """
    It dentifies indices of sites whose associated weights behave as outliers
    according to the z-score of the logarithmic norm of their weight vectors.
    
    Args:
        W (np.ndarray): Weight matrix (shape: q x Nv x Nh).
        z_score_threshold (float): Z-score threshold used to determine outliers. 
        sum_vis_dim (bool): If true, sums over the visible node's dimension.

    Returns:
        list: A list of indices where outliers were detected.
    
    """
    F = np.log(np.linalg.norm(W, axis=0))
    mean = np.mean(F, axis=0)
    std = np.std(F, axis=0)
    z_scores = ((F - mean)/ std) > z_score_threshold
    if sum_vis_dim:
        z_scores = z_scores.sum(axis=0)
    return (np.where(z_scores)[0]).tolist()


def create_outliers_dict(W: np.ndarray, z_score_threshold: float) -> dict :
    """
    Create a dictionary with the sites whoze associated weights behave as outilers
    for every hidden node.
    
    Args:
        W (np.ndarray): Weight matrix (shape: q x Nv x Nh).
        z_score_threshold (float): Z-score threshold used to determine outliers. 
    
    Returns:
        dict: A dictionary where the key is a hidden node index and the value is 
            a list with outliers indixes.
    """
    # creating dictionary
    _, _, Nh = W.shape
    outliers_list = []
    for a in range(Nh):
        Nv_outliers = find_outliers_index(W[:,:,a], z_score_threshold, False)
        outliers_list.append((int(a), Nv_outliers))
    return dict(outliers_list)

def k_unsqueeze(Wk: Tensor, 
                 n: int, 
                 k: int) -> Tensor:
    """
    A function to apply unsqueeze multiple times in the positions 1 and 0.
    
    Args:
        Wk (Tensor): Tensor to unsqueeze.
        n (int), k (int): Indicates how many times we will unsqueeze the tensor Wk.
    
    Returns:
        Tensor: an unsqueezed Tensor.    
    """
    
    Wk_unsqueezed = Wk.clone()
    for _ in range(n - (k + 1)):
        Wk_unsqueezed = Wk_unsqueezed.unsqueeze(1)
    for _ in range(k):
        Wk_unsqueezed = Wk_unsqueezed.unsqueeze(0)
    return Wk_unsqueezed


# Blume-Capel Model Analysis Functions
def couplings_zero(J: np.ndarray) -> np.ndarray:
    """
    Selects the couplings that should be zero in the Blume-Capel model from the 
    coupling tensor J of the RBM.

    Args:
        J (ndarray): Coupling tensor of shape (q, q, Nv, Nv).

    Returns:
        ndarray: Array of couplings that should be zero.
    """
    q,_,Nv,_ = J.shape
    couplings_zero = []
    for i in range(1, Nv):
        for j in range(i):
            couplings_zero.append(J[2,1,i,j])
            couplings_zero.append(J[2,0,i,j])
            couplings_zero.append(J[1,2,i,j])
            couplings_zero.append(J[0,2,i,j])
            couplings_zero.append(J[2,2,i,j])
    return np.array(couplings_zero)

def couplings_non_zero(J):
    """
    Selects the couplings that should be non-zero in the 1D Blume-Capel model from the 
    coupling tensor J of the RBM.
    
    Args:
        J (ndarray): Coupling tensor of shape (q, q, Nv, Nv).   
    
    Returns:    
        ndarray: Array of couplings that should be non-zero.
    """
    q,_,Nv,_ = J.shape
    couplings = []
    for i in range(1, Nv):
        for j in range(i):
            couplings.append(J[0,0,i,j])
            couplings.append(J[1,1,i,j])
            couplings.append(J[0,1,i,j])
            couplings.append(J[1,0,i,j])
    return np.array(couplings)

def root_mean_squared_error(J: np.ndarray, 
                            beta:float =0.2):
    """
    Computes the root mean squared error between the couplings in the 1D Blume-Capel model
    and the couplings inferred with the RBM.
    
    Args:
        J (ndarray): Coupling tensor of shape (q, q, Nv, Nv).
        beta (float): Coupling strength in the Blume-Capel model.
    
    Returns:
        float: Root mean squared error.
    """
    q,_,Nv,_ = J.shape
    error_list = []
    
    for i in range(1, Nv):
        for j in range(i):
            if (i - j) % Nv == 1:
                coupling_teo = beta
            else:
                coupling_teo = 0.0

            # zeros
            error_list.append((J[2,1,i,j] - 0.0)**2)
            error_list.append((J[2,0,i,j] - 0.0)**2)
            error_list.append((J[1,2,i,j] - 0.0)**2)
            error_list.append((J[0,2,i,j] - 0.0)**2)
            error_list.append((J[2,2,i,j] - 0.0)**2)

            # non-zeros
            error_list.append((J[0,0,i,j] - coupling_teo)**2)
            error_list.append((J[1,1,i,j] - coupling_teo)**2)
            error_list.append((J[0,1,i,j] + coupling_teo)**2)
            error_list.append((J[1,0,i,j] + coupling_teo)**2)
    
    return np.sqrt(np.array(error_list).mean())



