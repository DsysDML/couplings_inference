import numpy as np
import pandas as pd

# convert a matrix into a table
def matrix_to_table(F2):
    """
    Given a score matrix returns a table.    
    """
    Nv = F2.shape[0]
    data = []
    
    for i in range(Nv):
        for j in range(i + 1, Nv):
            data.append([i, j, F2[i, j]])
    
    return pd.DataFrame(data, columns=['i', 'j', 'F2'])

# creating a customized table
def custom_table(table, ascending=True, zero_index = True):
    """
    Creates custom table for scores and distances.

    paramenters:
    - table: table to transform in pandas format.
    - ascending: if true the score are dispose in ascending order, otherwise it will be descending.
    - zero_index: if true it is assumed that site index in the table start with 0. Otherwise, 
                 it is assumed that the starting index is 1.    
    """
    
    # selecting label for the score column
    score_name = table.columns[-1]
    rank_name = 'rank'
    
    # setting index columns as int type
    table['i'] = table['i'].astype(int)
    table['j'] = table['j'].astype(int)
    
    # changing from 1 to 0 start, if it's needed
    if not zero_index:
        table['i'] = table['i'] - 1
        table['j'] = table['j'] - 1
    
    # sorting the table according to the score 
    table = table.sort_values(by=score_name, ascending=ascending)
    
    # inserting column to account for the distance in the backbone structure
    table.insert(2, '|i-j|', table['j'] - table['i'])
    
    # reseting index
    table = table.reset_index(drop=True).reset_index(drop=False)
    table.set_index(['i', 'j'], inplace=True)
    table.columns = [rank_name, '|i-j|', score_name]
    table[rank_name] = table[rank_name].astype(int)
    
    return table # indexes: i, j ; cols: rank, |i-j|, score

# filter sequence distance
def filter_seq_distance(table, k = 0):
    """
    Given a (custom) table and a threshold k in the sequence distance it filers the rows such that |i-j| > k.
    
    Parameters:
    - table: custom tables.
    - k: threhshold in the site distance.
    """
    # filter
    table_copy = table[table['|i-j|'] > k].copy()
    table_copy['rank'] = pd.factorize(table_copy['rank'])[0] # reassingning the rakings
    return table_copy

def ROC_curve(xt, x, normalize=True):
    """
    Computes the Receiver Operating Characteristic curve and its area.
    
    Parameters:
    - xt: ground-truth array.
    - x: prediction array.
    - normalize: if true, the ouput is normalized.
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

def PPV_curve(xt, x):
    """
    Computes the Posituve Predictive Value curve and its area.
    
    Parameters:
    - xt: ground-truth array.
    - x: prediction array.
    """
    N = xt.size
    if np.isnan(x).sum() != 0:
        return np.zeros(N + 1), np.zeros(N + 1), np.nan
    
    sorted_indices = np.argsort(x)[::-1]

    xt = xt[sorted_indices]
    ppv, p = np.zeros(N), np.zeros(N)
    s = 0.0

    tp = 0
    for n in range(1, N+1):
        p[n-1] =  n
        ppv[n-1] = xt[:n].sum()/n
        s += ppv[n-1]
    
    return p, ppv, s