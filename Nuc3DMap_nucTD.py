import os,sys
import numpy as np
from math import isnan, sqrt
from scipy.stats import mannwhitneyu
from scipy.sparse import csr_matrix, eye, triu
from tqdm import tqdm
import pandas as pd
import pyBigWig as pybw
import seaborn as sns
import matplotlib.pyplot as plt
import pybedtools as pybt
import cooler
from numba import njit
from joblib import Parallel, delayed

@njit
def accumulate_diagonal(sum_arr, diag, shift_start, shift_end):
    n_diag = len(diag)
    n_sum = len(sum_arr)
    for offset in range(shift_start, shift_end + 1):
        for j in range(n_diag):
            pos = offset + j
            if pos < n_sum:
                sum_arr[pos] += diag[j]

def Get_Diamond_Matrix_Mean(data, size):
    N = data.shape[0]
    sum_arr = np.zeros(N, dtype=np.float64)
    
    # Iterate over diagonals
    # We need diagonals k where 1 <= k < 2*size
    for k in range(1, 2 * size):
        diag = data.diagonal(k)
        if len(diag) == 0:
            continue
            
        # Determine the range of i relative to r (row index) that this diagonal contributes to
        shift_start = max(0, k - size)
        shift_end = min(size - 1, k - 1)
        
        if shift_start > shift_end:
            continue
            
        # Vectorized addition with numba
        accumulate_diagonal(sum_arr, diag, shift_start, shift_end)

    # Calculate areas
    idxs = np.arange(N)
    lowerbounds = np.maximum(0, idxs - size + 1)
    upperbounds = np.minimum(idxs + size + 1, N)
    areas = (idxs + 1 - lowerbounds) * (upperbounds - idxs - 1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        means = sum_arr / areas
        
    means[areas == 0] = np.nan
    return means

def Which_Gap_Region(data):
    # ~1.5x faster than old
    n_bins = data.shape[0]

    upper = triu(data, format='csr')
    upper.sort_indices()

    M = np.full(n_bins, n_bins, dtype=np.int32)
    row_nnz = np.diff(upper.indptr)
    has_entries = row_nnz > 0
    rows_with = np.where(has_entries)[0]
    M[rows_with] = upper.indices[upper.indptr[rows_with]]

    S = np.minimum.accumulate(M[::-1])[::-1]

    gap_indices = []
    i = 0
    while i < n_bins:
        break_idx = S[i]
        gap_end = break_idx - 1
        if gap_end >= i + 1:
            gap_indices.extend(range(i, gap_end + 1))
            i = break_idx
        else:
            i += 1
    return np.array(gap_indices, dtype=int)

def Which_process_region(rmv_idx, n_bins, min_size):
    # ~8x faster
    proc_set = np.setdiff1d(np.arange(n_bins), rmv_idx)
    if len(proc_set) == 0:
        return dict()

    breaks = np.where(np.diff(proc_set) > 1)[0]
    run_starts = np.concatenate([[0], breaks + 1])
    run_ends = np.concatenate([breaks, [len(proc_set) - 1]])

    starts = proc_set[run_starts]
    ends = proc_set[run_ends]
    mask = (ends - starts) >= min_size

    proc_regions = {}
    for s, e in zip(starts[mask], ends[mask]):
        proc_regions[int(s)] = {'start': int(s), 'end': int(e)}
    return proc_regions

def Data_Norm(x, y):
    ret_x = np.cumsum(np.concatenate(([x[0]], np.diff(x) / np.abs(np.diff(x)).mean())))
    ret_y = np.cumsum(np.concatenate(([y[0]], np.diff(y) / np.abs(np.diff(y)).mean())))
    return ret_x, ret_y

def Change_Point(x, y):
    if len(x) != len(y):
        print("ERROR : The length of x and y should be the same")
        return 0
    n_bins = len(x)
    Fv = np.full(n_bins, np.nan)
    Ev = np.full(n_bins, np.nan)
    cp = [0]
    i = 0
    Fv[0] = 0
    while i < n_bins - 1:
        j = i + 1
        Fv[j] = sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2)
        while j < n_bins - 1:
            j = j + 1
            # k=(i+1):(j-1)
            Lj = sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2)
            Ev[j] = (np.abs((y[j] - y[i]) * x[(i + 1):j] - (x[j] - x[i]) * y[(i + 1):j] - (x[i] * y[j]) + (
                        x[j] * y[i]))).sum() / Lj
            # print Ev[j]
            # print x[(i+1):j]
            Fv[j] = Lj - Ev[j]
            #################################################
            # Not Original Code
            if isnan(Fv[j]) or isnan(Fv[j - 1]):
                j = j - 1
                cp.append(j)
                break
            ####################################################3
            if Fv[j] < Fv[j - 1]:
                j = j - 1
                cp.append(j)
                break
        i = j
    cp.append(n_bins - 1)
    return cp, Fv, Ev

def Detect_Local_Extreme(x):
    x = np.array(x)
    n_bins = len(x)
    ret = np.zeros(n_bins)
    x[np.isnan(x)] = 0
    if n_bins <= 3:
        ret[np.argmin(x)] = -1
        ret[np.argmax(x)] = 1
        return ret
    new_point_x, new_point_y = Data_Norm(x=np.arange(n_bins), y=x)
    x = new_point_y
    cp, _, _ = Change_Point(x=np.arange(n_bins), y=x)
    cp_len = len(cp)
    if cp_len <= 2:
        return ret
    for i in range(1, cp_len - 1):
        if x[cp[i]] >= x[cp[i] - 1] and x[cp[i]] >= x[cp[i] + 1]:
            ret[cp[i]] = 1
        else:
            if x[cp[i]] < x[cp[i] - 1] and x[cp[i]] < x[cp[i] + 1]:
                ret[cp[i]] = -1
        min_val = min(x[cp[i - 1]], x[cp[i]])
        max_val = max(x[cp[i - 1]], x[cp[i]])
        if np.min(x[cp[i - 1]:cp[i] + 1]) < min_val:
            ret[cp[i - 1] + np.argmin(x[cp[i - 1]:cp[i] + 1])] = -1
        if np.max(x[cp[i - 1]:cp[i] + 1]) > max_val:
            ret[cp[i - 1] + np.argmax(x[cp[i - 1]:cp[i] + 1])] = 1
    return ret

def scale(y):
    y = np.array(y)  # Ensure y is a numpy array
    return (y - np.mean(y)) / np.std(y, ddof=1)

def Get_Diamond_Matrix(data, i, size):
    n_bins = data.shape[0]
    new_mat = np.ones((size, size)) * np.NaN
    for k in range(1, size + 1):
        if i - (k - 1) >= 1 and i < n_bins:
            lower = min(i + 1, n_bins)
            upper = min(i + size, n_bins)
            new_mat[size - (k - 1) - 1, :upper - lower + 1] = data[i - (k - 1), lower - 1:upper]
    return new_mat[~np.isnan(new_mat)].tolist()

def Get_Upstream_Triangle(data, i, size):
    lower = max(0, i - size)
    tmp_mat = data[lower:i, lower:i]
    triag = np.triu(tmp_mat, k=1).flatten()
    return triag[triag != 0].tolist()

def Get_Downstream_Triangle(data, i, size):
    n_bins = data.shape[0]
    if i == n_bins:
        return np.NaN
    upperbound = min(i + size, n_bins)
    tmp_mat = data[i:upperbound, i:upperbound]
    triag = np.triu(tmp_mat, k=1).flatten()
    return triag[triag != 0].tolist()

def Get_Pvalue(data, size, scale):
    n_bins = data.shape[0]
    pvalue = np.ones(n_bins - 1)
    for i in range(1, n_bins):
        dia = Get_Diamond_Matrix(data, i, size)
        ups = Get_Upstream_Triangle(data, i, size)
        downs = Get_Downstream_Triangle(data, i, size)
        # Mann Whitney U Test
        wil_test = mannwhitneyu(x=np.array(dia) * scale, y=ups + downs, use_continuity=True, alternative='less')
        pvalue[i - 1] = wil_test.pvalue
    pvalue[np.isnan(pvalue)] = 1
    return pvalue

def Convert_Bin_To_Domain_TMP(n_bins, signal_idx, gap_idx, pvalues=None, pvalue_cut=None, rgmap=False):
    bins = dict()
    gap_rmv_idx = np.setdiff1d(np.arange(n_bins),gap_idx)
    gap_proc_region = Which_process_region(gap_rmv_idx, n_bins, min_size=0)
    for key in gap_proc_region:
        bins[gap_proc_region[key]['start']] = {'start': gap_proc_region[key]['start'], 'end': (gap_proc_region[key]['end']+1), 'score': 10, 'tag' : 'gap'}
    dom_rmv_idx = np.union1d(signal_idx, gap_idx)
    dom_proc_region = Which_process_region(dom_rmv_idx, n_bins, min_size=0)
    for key in dom_proc_region:
        bins[dom_proc_region[key]['start']] = {'start': dom_proc_region[key]['start'], 'end': (dom_proc_region[key]['end']+1), 'score': 10, 'tag' : 'domain'}
    bound_rmv_idx = np.setdiff1d(np.arange(n_bins),signal_idx)
    if rgmap:
        bound_proc_region = Which_process_region(bound_rmv_idx, n_bins, min_size=0)
    else:
        bound_proc_region = Which_process_region(bound_rmv_idx, n_bins, min_size=1)
    for key in bound_proc_region:
        bins[bound_proc_region[key]['start']] = {'start': bound_proc_region[key]['start'], 'end': (bound_proc_region[key]['end']+1), 'score': 10, 'tag' : 'boundary'}
    if pvalues is not None and pvalue_cut is not None:
        for key in bins:
            if bins[key]['tag'] == 'domain':
                start_id = bins[key]['start']
                end_id = bins[key]['end']
                p_value_constr = pvalues[start_id:end_id]
                bins[key]['score'] = p_value_constr.mean()
                p_value_constr = p_value_constr[p_value_constr < pvalue_cut]
                if end_id - start_id == len(p_value_constr):
                    bins[key]['tag'] = "boundary"
    return bins

def scale_mat(input_csr, window_size=5):
    # ~3x aster than old version
    from scipy.sparse import coo_matrix as coo_ctor
    n_bins = input_csr.shape[0]
    max_k = 2 * window_size

    repl_rows, repl_cols, repl_vals = [], [], []
    for k in range(1, max_k):
        diag_vals = input_csr.diagonal(-k).astype(np.float64)
        scaled = (diag_vals - np.mean(diag_vals)) / np.std(diag_vals, ddof=1)
        rows_k = np.arange(n_bins - k)
        repl_rows.append(rows_k)
        repl_cols.append(rows_k + k)
        repl_vals.append(scaled)

    repl_rows = np.concatenate(repl_rows)
    repl_cols = np.concatenate(repl_cols)
    repl_vals = np.concatenate(repl_vals)

    coo = input_csr.tocoo()
    offsets = coo.col - coo.row
    keep = ~((offsets >= 1) & (offsets < max_k))

    final_row = np.concatenate([coo.row[keep], repl_rows])
    final_col = np.concatenate([coo.col[keep], repl_cols])
    final_data = np.concatenate([coo.data[keep].astype(np.float64), repl_vals])

    return coo_ctor((final_data, (final_row, final_col)),
                     shape=(n_bins, n_bins)).tocsr()

def NucDom(bins_chr, csr_mat_chr, window_size, njobs = -1, statFilter=True):
    """

    :param hic_data: a list corresponding to the Hi-C data
    :param window_size: window size parameter for the NucDom algorithm
    :param True statFilter: whether to apply or not statistical filtering for false detection of TADs

    :returns: the :py:func:`list` of topologically associated domains, boundaries and gaps. Domains include the mean value
        of computed p-values by Wilcox Ranksum Test as score while boundaries and gaps have a score of zero.
    """
    n_bins = len(bins_chr)
    mean_cf = np.zeros(n_bins)
    pvalue = np.ones(n_bins)
    local_ext = np.ones(n_bins) * (-0.5)
    # Step 1: Get diamond matrix mean
    print("Get diamond matrix mean")
    mean_cf = Get_Diamond_Matrix_Mean(csr_mat_chr, window_size)
    # Step 2: Find Gap region
    print("Detect_Local_Extreme")
    gap_idx = Which_Gap_Region(csr_mat_chr)
    proc_regions = Which_process_region(rmv_idx=gap_idx, n_bins=n_bins, min_size=3)
    for key in tqdm(proc_regions):
        start = proc_regions[key]["start"]
        end = proc_regions[key]["end"]
        local_ext[start:end + 1] = Detect_Local_Extreme(mean_cf[start:end + 1])
    if statFilter:
        # Step 3: StatFilter
        print("StatFilter")
        csr_mat_chr_scale = scale_mat(csr_mat_chr,window_size)
        region_list = [(r['start'], r['end']) for r in proc_regions.values()]
        # pre-extract dense submatrices (shared by both old and new)
        dense_subs = [np.asarray(csr_mat_chr_scale[s:e + 1, s:e + 1].todense(), dtype=np.float64)
                  for s, e in region_list]
        results = Parallel(n_jobs=njobs)(
            delayed(Get_Pvalue)(sub, window_size, 1) for sub in tqdm(dense_subs, desc='parallel pvalue')
        )
        for (s, e), pv in zip(region_list, results):
            pvalue[s:e] = pv

        # for key in tqdm(proc_regions):
        #     start = proc_regions[key]['start']
        #     end = proc_regions[key]['end']
        #     pvalue[start:end] = Get_Pvalue(csr_mat_chr_scale[start:end + 1, start:end + 1].todense(), window_size, 1)

        local_ext[(local_ext == -1) & (pvalue < 0.05)] = -2
        local_ext[local_ext == -1] = 0
        local_ext[local_ext == -2] = -1
        pvalue_cut = 0.05
    else:
        pvalue = None
        pvalue_cut = None
    # Convert bin to domain
    print("Convert bin to domain")
    domains = Convert_Bin_To_Domain_TMP(n_bins=n_bins,
                                        signal_idx=np.where(local_ext == -1)[0],
                                        gap_idx=np.where(local_ext == -0.5)[0],
                                        pvalues=pvalue,
                                        pvalue_cut=pvalue_cut)
    return domains,mean_cf

def NucDom_RG(bins_chr, csr_mat_chr, window_size, statFilter=True):
    """

    :param hic_data: a list corresponding to the Hi-C data
    :param window_size: window size parameter for the NucDom algorithm
    :param True statFilter: whether to apply or not statistical filtering for false detection of TADs

    :returns: the :py:func:`list` of topologically associated domains, boundaries and gaps. Domains include the mean value
        of computed p-values by Wilcox Ranksum Test as score while boundaries and gaps have a score of zero.
    """
    n_bins = len(bins_chr)
    mean_cf = np.zeros(n_bins)
    pvalue = np.ones(n_bins)
    local_ext = np.ones(n_bins) * (-0.5)
    # Step 1: Get diamond matrix mean
    print("Get diamond matrix mean")
    mean_cf = Get_Diamond_Matrix_Mean(csr_mat_chr, window_size)
    mean_cf = np.nan_to_num(mean_cf, nan=0)
    # to preserve boundnaries
    bidx = bins_chr.loc[bins_chr['tag']==1,'arrayid']
    bmin = mean_cf[bidx].min()
    mean_cf[bidx] -= np.quantile(mean_cf[bidx],0.25)
    mean_cf[mean_cf <= 0] = bmin
    # Step 2: Find Gap region
    print("Find Gap region")
    gap_idx = bins_chr.loc[bins_chr['tag']==0,'arrayid'].values
    proc_regions = Which_process_region(rmv_idx=gap_idx, n_bins=n_bins, min_size=3)
    for key in tqdm(proc_regions):
        start = proc_regions[key]["start"]
        end = proc_regions[key]["end"]
        local_ext[start:end + 1] = Detect_Local_Extreme(mean_cf[start:end + 1])
    if statFilter:
        # Step 3: StatFilter
        print("StatFilter")
        csr_mat_chr_scale = scale_mat(csr_mat_chr,window_size)
        for key in tqdm(proc_regions):
            start = proc_regions[key]['start']
            end = proc_regions[key]['end']
            pvalue[start:end] = Get_Pvalue(csr_mat_chr_scale[start:end + 1, start:end + 1].todense(), window_size, 1)
        local_ext[(local_ext == -1) & (pvalue < 0.1)] = -2
        local_ext[local_ext == -1] = 0
        local_ext[local_ext == -2] = -1
        pvalue_cut = 0.1
    else:
        pvalue = None
        pvalue_cut = None
    # Convert bin to domain
    print("Convert bin to domain")
    domains = Convert_Bin_To_Domain_TMP(n_bins=n_bins,
                                        signal_idx=np.where(local_ext == -1)[0]+1,
                                        gap_idx=gap_idx,
                                        pvalues=pvalue,
                                        pvalue_cut=pvalue_cut,
                                        rgmap=True)
    return domains,mean_cf

def find_gaps(csr_mat, max_non_zeros=3, min_zeros=10):
    diagonal = csr_mat.diagonal()
    gaps = []
    start_gap = None
    non_zeros_count = 0
    zeros_count = 0
    for idx, value in enumerate(diagonal):
        if value == 0:
            zeros_count += 1
            if start_gap is None:
                start_gap = idx
            non_zeros_count = 0
        else:
            non_zeros_count += 1
            if non_zeros_count > max_non_zeros:
                if zeros_count >= min_zeros:
                    gaps.append((start_gap, idx - non_zeros_count))
                start_gap = None
                non_zeros_count = 0
                zeros_count = 0
    # If the last value is also within the tolerance and zeros count is valid, add the gap
    if start_gap is not None and zeros_count >= min_zeros:
        gaps.append((start_gap, len(diagonal) - 1))
    return gaps

def csr_coarsen(mat, block_size):
    """
    Efficiently aggregate the values in a CSR matrix using block sizes, focusing on non-zero elements.
    Adjusted for dimensions not perfectly divisible by block size.
    This is done by pooling *k*-by-*k* neighborhoods of pixels and aggregating.
    Each chromosomal block is coarsened individually. Avoids coarsening regions with successive zeros.

    Parameters:
    - mat: CSR matrix to be coarsened.
    - block_size: Size of block for coarsening.
    - zero_threshold: Number of successive zeros to avoid coarsening.

    Returns:
    - coarsened_mat: Coarsened CSR matrix with adjusted indices.
    """

    rows, cols = mat.nonzero()
    data = mat.data

    new_shape = (mat.shape[0] // block_size + (mat.shape[0] % block_size > 0),
                 mat.shape[1] // block_size + (mat.shape[1] % block_size > 0))

    aggregated_data = np.zeros(new_shape)
    print("Matrix Coarsening")
    for r, c, val in tqdm(zip(rows, cols, data)):
        new_r, new_c = r // block_size, c // block_size
        aggregated_data[new_r, new_c] += val

    coarsened_mat = csr_matrix(aggregated_data)
    return coarsened_mat

def bins_coarsen(bins_data, factor):
    """
    Coarsen the bins based on the provided factor.

    Parameters:
    - bins_data: DataFrame containing the original bins with columns 'chrom', 'start', and 'end'.
    - factor: Coarsening factor.

    Returns:
    - coarsened_bins: DataFrame containing the coarsened bins.
    """
    # List to store the coarsened bins
    coarsened_bins_list = []

    # Group by chromosome and apply coarsening independently for each chromosome
    for i in range(0, len(bins_data), factor):
        start_value = bins_data.iloc[i]['start']
        end_value = bins_data.iloc[min(i + factor - 1, len(bins_data) - 1)]['end']
        coarsened_bins_list.append((bins_data.iloc[i]['chrom'], start_value, end_value))

    coarsened_bins = pd.DataFrame(coarsened_bins_list, columns=['chrom', 'start', 'end'])
    return coarsened_bins

def preTADclr(clr, region, coarsen=0, tolerance=3, mingapsize=10, rgmap=False):
    """
    :param clr:
    :param region: chrom
    :param coarsen: int for coarsening
    :param gapsize: the size gap that need to be aware during coarsen
    :return:
    """
    try:
        bins_chr = clr.bins().fetch(region)
    except ValueError:
        print("Please use this format: chr1:1-100")
        exit(1)
    matsize = len(bins_chr)
    if rgmap:
        csr_mat_up = triu(clr.matrix(balance=False, sparse=True).fetch(region).tocsr())
        gap_bins = bins_chr.loc[bins_chr['tag']==0,['arrayid','arrayid']]
        gap_bins.columns = ['start','end']
    else:
        csr_mat_up = triu(clr.matrix(balance=True,sparse=True).fetch(region).tocsr())
        gaps = find_gaps(csr_mat_up, max_non_zeros=tolerance, min_zeros=mingapsize)
        gap_bins = pd.DataFrame({
            'start':[gap[0] for gap in gaps],
            'end':[gap[1] for gap in gaps]
        })
    if coarsen > 0:
        print(f"Coarsen with factor {coarsen}")
        csr_mat_up_proc = csr_coarsen(csr_mat_up, coarsen)
        matsize_proc = csr_mat_up_proc.shape[0]
        bins_chr_proc = bins_coarsen(bins_chr, coarsen)
    else:
        csr_mat_up_proc = csr_mat_up
        bins_chr_proc = bins_chr
        matsize_proc = matsize
    csr_mat = csr_matrix(csr_mat_up_proc.shape, dtype=np.float32)
    # make to mat into full matrix format
    csr_mat = csr_mat + csr_mat_up_proc + csr_mat_up_proc.transpose()
    csr_mat = csr_mat - eye(matsize_proc).multiply(csr_mat.diagonal() / 2)
    return bins_chr_proc, csr_mat, gap_bins

def integrateDomainGaps(result_df,gaps_df,bins_chr,leastnuc=2,rgmap=False):
    # Utility function to fill numpy arrays based on DataFrame columns
    def fill_array_from_df(arr, start, end, value):
        arr[start:end + 1] = value
    # Define a function to check if a group labeled as 'domain' has at least two rows where nuc == 1
    def validate_domain(group):
        if group['final_tag'].iloc[0] == 'domain' and group['nuc'].sum() < leastnuc:
            group['final_tag'] = 'gap'
            group['final_score'] = 1
        return group
    max_end_value = bins_chr['idx'].values[-1]
    tmp_df = pd.DataFrame({
        'index': range(0, max_end_value + 1),
        'tag_result': [None] * (max_end_value + 1),
        'tag_gap': [None] * (max_end_value + 1),
        'score_result': [None] * (max_end_value + 1),
        'score_gap': [None] * (max_end_value + 1)
    })
    # Use apply method to fill arrays
    tag_result_arr = np.array([None] * (max_end_value + 1), dtype=object)
    score_result_arr = np.array([None] * (max_end_value + 1), dtype=object)
    tag_gap_arr = np.array([None] * (max_end_value + 1), dtype=object)
    score_gap_arr = np.array([None] * (max_end_value + 1), dtype=object)
    # fill four arrays
    result_df.apply(lambda row: fill_array_from_df(tag_result_arr, row['start'], row['end'], row['tag']), axis=1)
    result_df.apply(lambda row: fill_array_from_df(score_result_arr, row['start'], row['end'], row['score']), axis=1)
    gaps_df.apply(lambda row: fill_array_from_df(tag_gap_arr, row['start'], row['end'], row['tag']), axis=1)
    gaps_df.apply(lambda row: fill_array_from_df(score_gap_arr, row['start'], row['end'], row['score']), axis=1)
    tmp_df['tag_result'] = tag_result_arr
    tmp_df['tag_gap'] = tag_gap_arr
    tmp_df['score_result'] = score_result_arr
    tmp_df['score_gap'] = score_gap_arr
    # Generate final_tag and final_score columns
    tmp_df['final_tag'] = tmp_df['tag_gap'].combine_first(tmp_df['tag_result']).fillna('gap')
    tmp_df['final_score'] = pd.to_numeric(tmp_df['score_gap'].combine_first(tmp_df['score_result']), errors='coerce')
    tmp_df['final_score'].fillna(1,inplace=True)
    # Group by consecutive rows with the same tag and calculate the start, end and mean score
    tmp_df['group'] = (tmp_df['final_tag'] != tmp_df['final_tag'].shift()).cumsum()
    if not rgmap:
        tmp_df['nuc'] = bins_chr['nuc'].values
        # Apply the validate_domain function to tmp_df
        tmp_df = tmp_df.groupby('group',group_keys=True).apply(validate_domain)
    # Regroup and calculate the start, end, and mean score
    tmp_df['group'] = (tmp_df['final_tag'] != tmp_df['final_tag'].shift()).cumsum()
    tmp_df.reset_index(drop=True,inplace=True)
    final_df = tmp_df.groupby('group').agg(
        start=('index', 'first'),
        end=('index', 'last'),
        tag=('final_tag', 'first'),
        mean_score=('final_score', 'mean')
    ).reset_index(drop=True)
    return final_df

def optResults(final_df,boundarycut=1000):
    # small gaps are consider boudnary
    final_df.loc[
        ((final_df['end'] - final_df['start']) <= boundarycut) & (final_df['tag'] == 'gap'), 'tag'] = 'boundary'
    final_df['group'] = (final_df['tag'] != final_df['tag'].shift()).cumsum()
    # Aggregate rows
    aggregated_df = final_df.groupby(['group', 'tag']).agg({
        'chrom':'first',
        'start': 'min',
        'end': 'max',
        'score': 'mean',
        'snucidx' : 'min',
        'enucidx' : 'max'
    }).reset_index()
    final_df = aggregated_df[['chrom','start', 'end', 'score', 'tag', 'snucidx','enucidx']]
    final_df.loc[(final_df['tag']=='domain')&(final_df['score']>-1),'score'] = -1
    return final_df

def callTADs(ret, gaps, bins_chr,rgmap=False):
    result = {'start': [], 'end': [], 'score': [], 'tag': []}
    for key in sorted(ret):
        result['tag'].append(ret[key]['tag'])
        result['start'].append(ret[key]['start'])
        result['end'].append(ret[key]['end'])
        if ret[key]['tag'] == 'domain':
            result['score'].append(ret[key]['score'])
        else:
            result['score'].append(0)
    # inherit from pytadbit line 101
    max_score = max(result['score'])
    for i in range(len(result['score'])):
        if max_score != 0:
            result['score'][i] = 1 - int((result['score'][i] / max_score) * 10)
        else:
            max_score = 1
            result['score'][i] = 1 - int((result['score'][i] / max_score) * 10)
    result_df = pd.DataFrame(result)
    # convert to nucleosome-bin based idx
    result_df['end'] -= 1
    bins_chr['idx'] = np.arange(len(bins_chr))
    # refine by gaps; should be in same chrom
    # add score and tag to gaps
    gaps_df = gaps.copy()
    gaps_df['tag'] = 'gap'
    gaps_df['score'] = 1
    print("Finalize results")
    merged_df = integrateDomainGaps(result_df, gaps_df, bins_chr, rgmap=rgmap)
    startidx_dict = dict(zip(bins_chr['idx'],bins_chr['start']))
    endidx_dict = dict(zip(bins_chr['idx'],bins_chr['end']))
    merged_df['start_coor'] = merged_df['start'].map(startidx_dict)
    merged_df['end_coor'] = merged_df['end'].map(endidx_dict)
    merged_df['chrom'] = bins_chr['chrom'].values[0]
    merged_df = merged_df[['chrom','start_coor','end_coor','mean_score','tag','start','end']]
    merged_df.columns = ['chrom','start','end','score','tag','snucidx','enucidx']
    if rgmap:
        merged_df['snucidx'] = merged_df['snucidx'].map(dict(zip(bins_chr['idx'], bins_chr['snucidx'])))
        merged_df['enucidx'] = merged_df['enucidx'].map(dict(zip(bins_chr['idx'], bins_chr['enucidx'])))
    merged_df = optResults(merged_df)
    return merged_df

def makeConsecutiveCoor(nucdom_df,chroms,shiftsize=0):
    nucdom_df_adj_list = []
    nucdom_df_gb = nucdom_df.groupby('chrom')
    for chrom in chroms:
        print(chrom)
        chrom_df = nucdom_df_gb.get_group(chrom)
        if len(chrom_df) > 3:
            cur_domain_idx = chrom_df[chrom_df['tag'] == 'domain'].index.values
            cur_gb_idx = chrom_df[chrom_df['tag'].isin(['gap', 'boundary'])].index.values
            cur_gb_adjacent_d_end_idx = np.sort(list(
                set(np.concatenate([cur_domain_idx - 1])).intersection(set(cur_gb_idx))))  # idx 'end' need to change
            cur_gb_adjacent_d_start_idx = np.sort(list(
                set(np.concatenate([cur_domain_idx + 1])).intersection(set(cur_gb_idx))))  # idx 'start' need to change
            cur_gb_new_end = chrom_df.loc[cur_gb_adjacent_d_end_idx[1:] + 1, 'start'].values
            cur_gb_new_start = chrom_df.loc[cur_gb_adjacent_d_start_idx[:-1] - 1, 'end'].values
            # adjust all except first end and last start
            chrom_df.loc[cur_gb_adjacent_d_start_idx[:-1], 'start'] = cur_gb_new_start
            chrom_df.loc[cur_gb_adjacent_d_end_idx[1:], 'end'] = cur_gb_new_end
            # make up the first end
            chrom_df.loc[cur_gb_adjacent_d_end_idx[0], 'end'] = chrom_df.iloc[1, :]['start']
            # make up the last start
            chrom_df.loc[cur_gb_adjacent_d_start_idx[-1], 'start'] = chrom_df.iloc[-2, :]['end']
            # make up two adjacent boundary and gap, keep boundary coordinate
            chrom_df['shift_start'] = [chrom_df['start'].values[0]] + chrom_df['end'].values[:-1].tolist()
            nm_b_idx = chrom_df[
                (chrom_df['shift_start'] != chrom_df['start']) & (chrom_df['tag'] == 'boundary')].index.values
            nm_g_idx = chrom_df[
                (chrom_df['shift_start'] != chrom_df['start']) & (chrom_df['tag'] == 'gap')].index.values
            chrom_df.loc[nm_b_idx - 1, 'end'] = chrom_df.loc[nm_b_idx, 'start'].values
            chrom_df.loc[nm_g_idx, 'start'] = chrom_df.loc[nm_g_idx, 'shift_start'].values
            # check consecutive
            chrom_df['shift_start'] = [chrom_df['start'].values[0]] + chrom_df['end'].values[:-1].tolist()
            if len(chrom_df[chrom_df['shift_start'] != chrom_df['start']]) != 0:
                # force consecutive
                chrom_df.loc[chrom_df['shift_start'] != chrom_df['start'],'start'] = \
                    chrom_df.loc[chrom_df['shift_start'] != chrom_df['start'],'shift_start'].values
            # check if start < end
            idx = chrom_df[chrom_df['start']>chrom_df['end']].index.values
            if len(idx) != 0:
                print('Rare case, please report')
                # make up
                chrom_df.loc[idx,'start'] = chrom_df.loc[idx,'end'].values
            chrom_df.drop('shift_start', axis=1, inplace=True)
            nucdom_df_adj_list.append(chrom_df)
        else:
            nucdom_df_adj_list.append(chrom_df)
    nucdom_df_adj = pd.concat(nucdom_df_adj_list)
    # shift a certain empirical size (normally 150 bp)
    boundary_idx = nucdom_df_adj[nucdom_df_adj['tag']=='boundary'].index.values
    nucdom_df_adj.iloc[boundary_idx, 1] -= shiftsize
    nucdom_df_adj.iloc[boundary_idx, 2] -= shiftsize
    nucdom_df_adj.iloc[boundary_idx-1, 2] -= shiftsize
    nucdom_df_adj.iloc[boundary_idx+1, 1] -= shiftsize
    return nucdom_df_adj

def TransferLearning(targeted_tads_df, h1_nucdom_f, factor_pairs, targeted_clr):
    """
    Perform transfer learning from H1 nucleosome domains to a targeted dataset
    using multiple factors.

    Parameters:
      targeted_tads_df (pd.DataFrame): DataFrame of the targeted nucleosome domains
          (e.g. GM12878) with at least the columns ["chrom", "start", "end", "tag", ...].
      h1_nucdom_f (str): File path for the H1 nucleosome domain file.
      factor_pairs (list): List of tuples, each containing (h1_factor_bw, targeted_factor_bw).
          Example: [('H1_CTCF.bw', 'GM_CTCF.bw'), ('H1_Pol2.bw', 'GM_Pol2.bw')]
          Boundaries are prioritized by order (first factor > second > ...).
      targeted_clr: Cooler object for the targeted dataset.

    Returns:
      nucdom_trans (pd.DataFrame): The targeted nucleosome domain DataFrame with transferred boundaries.
    """

    # --- Load H1 nucleosome domain and keep boundaries ---
    h1_nucdom = pd.read_table(h1_nucdom_f)
    h1_nucb = h1_nucdom[h1_nucdom['tag'] == 'boundary'].copy()
    h1_nucb['mid'] = h1_nucb[['start', 'end']].mean(axis=1).astype(int)

    # helper function to find an elbow point (signal cutoff)
    def findSignalCut(vals, outname=None, density_cut=0.0015, xmin=10):
        # plot density and find elbow point
        ax = sns.kdeplot(vals)
        _points = ax.get_lines()[0].get_data()
        if outname is not None:
            plt.savefig(outname, dpi=300)
            plt.close()
        else:
            plt.close()
        x = _points[0]
        y = _points[1]

        x_ymax = np.where(y == y.max())[0][0]
        x_ycut = np.where(y <= density_cut)[0]
        # take the first value after the max
        xcut = x_ycut[x_ycut > x_ymax][0]
        xcut = np.max([xcut, xmin])
        return xcut

    # --- Prepare targeted nucleosome domains ---
    targeted_nucdom = targeted_tads_df.copy()
    targeted_nucb = targeted_nucdom[targeted_nucdom['tag'] == 'boundary'].copy()

    # Accumulate all boundaries to transfer (from all factors)
    all_trans_boundaries = pd.DataFrame(columns=['chrom', 'start', 'end'])
    extsize = 100

    # --- Process each factor pair ---
    for factor_idx, (h1_factor_f, targeted_factor_f) in enumerate(factor_pairs):
        print(f"\n=== Processing factor {factor_idx + 1}: {h1_factor_f} / {targeted_factor_f} ===")

        h1_factor_bw = pybw.open(h1_factor_f)
        targeted_factor_bw = pybw.open(targeted_factor_f)

        h1_signal = []
        targeted_signal = []

        for idx, row in tqdm(h1_nucb.iterrows(), desc=f"Processing H1 boundaries (factor {factor_idx + 1})"):
            h1_signal.append(h1_factor_bw.stats(row['chrom'], row['start'] - extsize, row['end'] + extsize,
                                                type='max', exact=True)[0])
            targeted_signal.append(targeted_factor_bw.stats(row['chrom'], row['start'] - extsize, row['end'] + extsize,
                                                             type='max', exact=True)[0])

        h1_factor_bw.close()
        targeted_factor_bw.close()

        # Create temporary DataFrame for this factor
        h1_nucb_temp = h1_nucb.copy()
        h1_nucb_temp['h1_signal'] = h1_signal
        h1_nucb_temp['targeted_signal'] = targeted_signal

        # Find cutoffs
        h1_cut = findSignalCut(h1_nucb_temp['h1_signal'].values, density_cut=0.015)
        targeted_cut = findSignalCut(h1_nucb_temp['targeted_signal'].values)

        h1_nucb_temp['h1_binary'] = np.where(h1_nucb_temp['h1_signal'] >= h1_cut, 1, 0)
        h1_nucb_temp['targeted_binary'] = np.where(h1_nucb_temp['targeted_signal'] >= targeted_cut, 1, 0)

        # Select candidate transfer boundaries (high signal in both)
        candidates = h1_nucb_temp[h1_nucb_temp[['h1_binary', 'targeted_binary']].sum(axis=1) == 2].copy()

        if len(candidates) == 0:
            print(f"No candidate boundaries found for factor {factor_idx + 1}")
            continue

        print(f"Factor {factor_idx + 1}: {len(candidates)} initial candidates")

        # Remove candidates that overlap with existing targeted boundaries
        if len(targeted_nucb) > 0:
            inter_with_targeted = pybt.BedTool.intersect(
                pybt.BedTool.from_dataframe(candidates[['chrom', 'start', 'end']]),
                pybt.BedTool.from_dataframe(targeted_nucb[['chrom', 'start', 'end']]),
                u=True
            )
            if len(inter_with_targeted) > 0:
                olp_with_targeted = inter_with_targeted.to_dataframe()
                candidates = pd.merge(candidates, olp_with_targeted, how='left', indicator=True) \
                    .query('_merge == "left_only"') \
                    .drop('_merge', axis=1)

        print(f"Factor {factor_idx + 1}: {len(candidates)} candidates after removing overlap with existing boundaries")

        # Remove candidates that overlap with previously selected transfer boundaries (prioritize earlier factors)
        if len(all_trans_boundaries) > 0 and len(candidates) > 0:
            inter_with_prev = pybt.BedTool.intersect(
                pybt.BedTool.from_dataframe(candidates[['chrom', 'start', 'end']]),
                pybt.BedTool.from_dataframe(all_trans_boundaries[['chrom', 'start', 'end']]),
                u=True
            )
            if len(inter_with_prev) > 0:
                olp_with_prev = inter_with_prev.to_dataframe()
                candidates = pd.merge(candidates, olp_with_prev, how='left', indicator=True) \
                    .query('_merge == "left_only"') \
                    .drop('_merge', axis=1)

        print(f"Factor {factor_idx + 1}: {len(candidates)} boundaries to transfer (after priority filtering)")

        # Add to cumulative list
        if len(candidates) > 0:
            all_trans_boundaries = pd.concat([all_trans_boundaries, candidates[['chrom', 'start', 'end']]])

    print(f"\n=== Total boundaries to transfer: {len(all_trans_boundaries)} ===")

    if len(all_trans_boundaries) == 0:
        print("No boundaries to transfer, returning original data")
        return targeted_nucdom

    # Use all_trans_boundaries as the boundaries to transfer
    h1_nucb_trans_noolp = all_trans_boundaries.copy()

    # --- Define helper functions for splitting/merging regions ---
    def split_and_merge_adjust_rows_with_tags(row):
        results = []
        chrom, start, end, id_val, snucidx, enucidx, score, tag = (
            row['chrom'], row['start'], row['end'], row['id'], row['snucidx'], row['enucidx'], row['score'], row['tag']
        )
        start_x, end_x, snucidx_x, enucidx_x = row['start_x'], row['end_x'], row['snucidx_x'], row['enucidx_x']

        sub_rows = []
        if start_x < start and end_x <= end:
            sub_rows.append([chrom, end_x, end, id_val, enucidx_x, enucidx, score, tag])
        if start_x <= end and end_x > end:
            sub_rows.append([chrom, start, start_x, id_val, snucidx, snucidx_x, score, tag])
        if start_x >= start and end_x <= end:
            sub_rows.append([chrom, start, start_x, id_val + 0.1, snucidx, snucidx_x, score, tag])
            sub_rows.append([chrom, start_x, end_x, id_val + 0.2, snucidx_x, enucidx_x, 1, 'boundary'])
            sub_rows.append([chrom, end_x, end, id_val + 0.3, enucidx_x, enucidx, score, tag])

        if len(sub_rows) == 3:
            # merge first and second if possible
            if sub_rows[0][4] == sub_rows[1][4]:
                sub_rows[1][1] = sub_rows[0][1]
                sub_rows[1][4] = sub_rows[0][4]
                sub_rows.pop(0)
            # merge second and third if possible
            if len(sub_rows) > 2 and sub_rows[1][5] == sub_rows[2][5]:
                sub_rows[1][2] = sub_rows[2][2]
                sub_rows[1][5] = sub_rows[2][5]
                sub_rows.pop(2)
        results.extend(sub_rows)
        return results

    def multib_split(group):
        sorted_group = group.sort_values(by=['start_x', 'end_x'])
        results = []
        chrom = group['chrom'].iloc[0]
        main_start = group['start'].iloc[0]
        main_end = group['end'].iloc[0]
        id_base = group['id'].iloc[0]
        snucidx = group['snucidx'].iloc[0]
        enucidx = group['enucidx'].iloc[0]
        score = group['score'].iloc[0]
        tag = group['tag'].iloc[0]
        current_start = main_start
        segment_count = 0

        for _, sub in sorted_group.iterrows():
            start_x = max(current_start, sub['start_x'])
            end_x = min(main_end, sub['end_x'])

            if current_start < start_x:
                segment_count += 1
                new_id = id_base + segment_count * 0.001
                results.append([chrom, current_start, start_x, new_id, snucidx, sub['snucidx_x'], score, tag])

            segment_count += 1
            new_id = id_base + segment_count * 0.001
            results.append([chrom, start_x, end_x, new_id, sub['snucidx_x'], sub['enucidx_x'], 1, 'boundary'])
            snucidx = sub['enucidx_x']
            current_start = end_x

        if current_start < main_end:
            segment_count += 1
            new_id = id_base + segment_count * 0.001
            results.append([chrom, current_start, main_end, new_id, snucidx, enucidx, score, tag])

        return results

    # --- Insert transferred boundaries into the targeted nucleosome domains ---
    # Create a cooler object for the targeted species.
    # (Here we use a fixed file path; consider parameterizing this if needed.)
    chroms = targeted_clr.chromnames

    chrom_list = []
    # Group the transferred boundaries and the targeted domains by chromosome
    h1_nucb_trans_noolp_gb = h1_nucb_trans_noolp.groupby('chrom')
    targeted_nucdom_gb = targeted_nucdom.groupby('chrom')

    for chrom in chroms:
        print(f"Processing {chrom}")
        try:
            chrom_df = targeted_nucdom_gb.get_group(chrom).copy()
        except KeyError:
            continue

        chrom_df['id'] = np.arange(len(chrom_df))
        bins_chr = targeted_clr.bins().fetch(chrom)
        bins_chr['nucidx'] = np.arange(len(bins_chr))

        try:
            trans_b_chr = h1_nucb_trans_noolp_gb.get_group(chrom)
        except KeyError:
            print('No transferred boundaries on this chromosome, skipping...')
            chrom_df.drop('id', axis=1, inplace=True)
            chrom_list.append(chrom_df)
            continue

        print("Mapping transferred boundaries")
        # Map transferred boundaries to nucleosome bins
        trans_b_bins = pybt.BedTool.intersect(
            pybt.BedTool.from_dataframe(trans_b_chr[['chrom', 'start', 'end']]),
            pybt.BedTool.from_dataframe(bins_chr[['chrom', 'start', 'end', 'nucidx']]),
            wa=True, wb=True
        ).to_dataframe(disable_auto_names=True,
                       names=['chrom', 'start', 'end', 'chrom_x', 'start_x', 'end_x', 'nucidx'])

        trans_b_bins_aggregated = trans_b_bins.groupby(['chrom', 'start', 'end']).agg(
            snucidx=pd.NamedAgg(column='nucidx', aggfunc='min'),
            enucidx=pd.NamedAgg(column='nucidx', aggfunc='max')
        ).reset_index()
        nucid2start = dict(zip(bins_chr['nucidx'], bins_chr['start']))
        nucid2end = dict(zip(bins_chr['nucidx'], bins_chr['end']))
        trans_b_bins_aggregated['nuc_start'] = trans_b_bins_aggregated['snucidx'].map(nucid2start)
        trans_b_bins_aggregated['nuc_end'] = trans_b_bins_aggregated['enucidx'].map(nucid2end)
        trans_b_bins_aggregated = trans_b_bins_aggregated[['chrom', 'nuc_start', 'nuc_end', 'snucidx', 'enucidx']]
        trans_b_bins_aggregated.columns = ['chrom', 'start', 'end', 'snucidx', 'enucidx']
        del nucid2start, nucid2end

        # find interval overlap with transferred boundaries
        inter_res = pybt.BedTool.intersect(
            pybt.BedTool.from_dataframe(chrom_df),
            pybt.BedTool.from_dataframe(trans_b_bins_aggregated),
            wa=True, wb=True
        ).to_dataframe(disable_auto_names=True,
                       names=list(chrom_df.columns) + ['chrom_x', 'start_x', 'end_x', 'snucidx_x', 'enucidx_x'])
        inter_res.drop_duplicates(inplace=True)
        # remove domains totally contained in boundary
        inter_res = inter_res[
            ~((inter_res['start'] >= inter_res['start_x']) & (inter_res['end'] <= inter_res['end_x']))]
        print("Inserting transferred boundaries")
        # unchanged regions
        chrom_df_unchange = chrom_df[~chrom_df['id'].isin(inter_res['id'].values)]
        # separate domain and background regions
        inter_res_d = inter_res[inter_res['tag'] == 'domain']
        inter_res_bg = inter_res[inter_res['tag'] != 'domain']
        # Process splitting for domain regions
        split_inter_res_d_list = []
        for group, group_df in inter_res_d.groupby(['chrom', 'start', 'end']):
            if len(group_df) == 1:
                split_inter_res_d_list.extend(split_and_merge_adjust_rows_with_tags(group_df.iloc[0]))
            else:
                split_inter_res_d_list.extend(multib_split(group_df))
        split_inter_res_d = pd.DataFrame(split_inter_res_d_list,
                                         columns=['chrom', 'start', 'end', 'id', 'snucidx', 'enucidx', 'score', 'tag'])
        chrom_df_modi = pd.concat([chrom_df_unchange,
                                   split_inter_res_d,
                                   inter_res_bg[chrom_df.columns].drop_duplicates()]).sort_values('id')
        # make consecutive regions
        chrom_df_modi['shift_start'] = [chrom_df_modi['start'].values[0]] + chrom_df_modi['end'].values[:-1].tolist()
        chrom_df_modi.reset_index(drop=True, inplace=True)
        nm_domain_idx = chrom_df_modi[(chrom_df_modi['shift_start'] != chrom_df_modi['start']) &
                                      (chrom_df_modi['tag'] == 'domain')].index.values
        chrom_df_modi.loc[nm_domain_idx - 1, 'end'] = chrom_df_modi.loc[nm_domain_idx, 'start'].values
        nm_bg_idx = chrom_df_modi[(chrom_df_modi['shift_start'] != chrom_df_modi['start']) &
                                  (chrom_df_modi['tag'] != 'domain')].index.values
        chrom_df_modi.loc[nm_bg_idx, 'start'] = chrom_df_modi.loc[nm_bg_idx - 1, 'end'].values
        chrom_df_modi.drop(['id', 'shift_start'], axis=1, inplace=True)
        chrom_df_modi = chrom_df_modi[chrom_df_modi['start'] != chrom_df_modi['end']]
        chrom_list.append(chrom_df_modi)

    nucdom_trans = pd.concat(chrom_list)
    return nucdom_trans

