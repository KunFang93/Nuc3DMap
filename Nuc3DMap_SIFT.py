import numpy as np
from scipy import sparse
import math
from multiprocessing import Process, Manager
from scipy.stats import expon
from scipy.ndimage import gaussian_filter
from statsmodels.stats.multitest import multipletests
from scipy.ndimage import maximum_filter
import scipy.ndimage.measurements as scipy_measurements
from numba import njit
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

# Inspired from MUSTCHE, Mainbody of SIFT stem from it
@njit
def find_chunks(bins_end,distance_in_bp,chunk_coef):
    # find the index for chunking
    chunk_start_indices,chunk_end_indices = [],[]
    start_chunk_count,end_chunk_count = 0,0
    for i, value in enumerate(bins_end):
        if value >= ((chunk_coef-1)+chunk_coef*start_chunk_count)*distance_in_bp:
            chunk_start_indices.append(i)
            start_chunk_count += 1
        if value >= (chunk_coef+chunk_coef*end_chunk_count)*distance_in_bp:
            chunk_end_indices.append(i)
            end_chunk_count += 1
    # dealing very large bin at the beginning
    if chunk_start_indices[0] == 0 and chunk_end_indices[0] == 0:
        chunk_start_indices = chunk_start_indices
        chunk_end_indices = chunk_end_indices[1:] + [len(bins_end) - 1]
    elif chunk_start_indices[0] != 0 and chunk_end_indices[0] != 0:
        chunk_start_indices = [0] + chunk_start_indices
        chunk_end_indices = chunk_end_indices + [len(bins_end)-1]
    else:
        print("Find chunk error, please report to author")
    return chunk_start_indices,chunk_end_indices

@njit
def find_distance_in_px(rows,cols,midpoints,distance_in_bp):
    rows_mid = midpoints[rows]
    cols_mid = midpoints[cols]
    # find the upper triangle px
    distances = cols_mid-rows_mid
    rows_inrange = rows[distances<distance_in_bp]
    cols_inrange = cols[distances<distance_in_bp]
    return max(cols_inrange-rows_inrange)

def read_cooler_varbins(clr, distance_in_bp,chrom, chunk_coef=5, intensity_cut=0.2):
    """
    :param f: .cool file path
    :param chr: Which chromosome to read the file for
    :return: Numpy matrix of contact counts
    """
    bins = clr.bins().fetch(chrom)[:].reset_index(drop=True)
    # find the index for chunking
    chunk_start_indices, chunk_end_indices = find_chunks(bins['end'].values,distance_in_bp,chunk_coef=chunk_coef)
    # make sure the chunk boundaries are paired
    if len(chunk_start_indices) != len(chunk_end_indices):
        chunk_start_indices = chunk_start_indices[:-1]
        if len(chunk_start_indices) != len(chunk_end_indices):
            print("read_cool error, situation not consider, please contact author")
            exit(1)
    result = []
    for i in range(len(chunk_start_indices)):
        start_idx = chunk_start_indices[i]
        end_idx = chunk_end_indices[i]
        start = bins.loc[start_idx,'start']
        end = bins.loc[end_idx,'end']
        print(start, end)
        temp = clr.matrix(balance=False, sparse=True).fetch((chrom, int(start), int(end)))
        # remove potential noisy
        temp_csr = temp.tocsr()
        # Replace values less than 0.2 with 0
        temp_csr.data[temp_csr.data < intensity_cut] = 0
        # Convert back to coo_matrix if needed
        temp = temp_csr.tocoo()
        temp = sparse.triu(temp)
        np.nan_to_num(temp, copy=False, nan=0, posinf=0, neginf=0)
        if result == []:
            result += [list(start_idx + temp.row), list(start_idx + temp.col), list(temp.data)]
            prev_block = set(
                [(x, y, v) for x, y, v in zip(start_idx + temp.row, start_idx + temp.col, temp.data)])
        else:
            cur_block = set(
                [(x, y, v) for x, y, v in zip(start_idx + temp.row, start_idx + temp.col, temp.data)])
            to_add_list = list(cur_block - prev_block)
            del prev_block
            result[0] += [x[0] for x in to_add_list]
            result[1] += [x[1] for x in to_add_list]
            result[2] += [x[2] for x in to_add_list]
            prev_block = cur_block
            del cur_block
    # raise NameError('Reading from the file failed!')
    if len(result) == 0:
        print(f'There is no contact in chrmosome {chrom} to work on.')
        return [], [], [], chunk_start_indices, chunk_end_indices
    x = np.array(result[0])
    y = np.array(result[1])
    val = np.array(result[2])
    ##########################
    if len(val) == 0:
        print(f'There is no contact in chrmosome {chrom} to work on.')
        return [], [], [], chunk_start_indices, chunk_end_indices
    else:
        val[np.isnan(val)] = 0
    midpoints = ((bins['start'] + bins['end'])/2).values
    dist_f = np.logical_and(np.abs(midpoints[x] - midpoints[y]) <= distance_in_bp, val > 0)
    x = x[dist_f]
    y = y[dist_f]
    val = val[dist_f]
    # return np.array(x),np.array(y),np.array(val), res, normVec
    if len(val > 0):
        return np.array(x), np.array(y), np.array(val), chunk_start_indices, chunk_end_indices
    else:
        print(f'There is no contact in chrmosome {chrom} to work on.')
        return [], [], [], chunk_start_indices, chunk_end_indices
@njit
def compute_gaussian_params(o, i, s):
    # The calculated σ controls the degree of smoothing,
    # the kernel width w determines the size of the area affected by the filter,
    # and the truncate factor t adjusts the kernel's effective range to optimize the filter's performance
    sigma = o * 2 ** ((i - 1) / s)
    w = 2 * math.ceil(2 * sigma) + 1
    t = ((w - 1) / 2 - 0.5) / sigma
    return sigma, t

def SIFT(c, start, end, distance_in_px, octave_values, st, pt, ignore_diags=2, fp_size=5):
    print("Prepare contact matrix")
    # SIFT algorithm actually...
    # only consider loops over certain diagonals
    nz = np.logical_and(c != 0, np.triu(c, ignore_diags))
    if np.sum(nz) < 50:
        print("The non-zeros in the contact matrix is too less, cannot find significant interactions")
        return []
    # assign all values below certain diagonals to 2, remove too closed interaction?
    c[np.tril_indices_from(c, ignore_diags)] = 2
    # assign all value above distance_filter to 2, why 2, guess log2 convert it 1
    c[np.triu_indices_from(c, k=(distance_in_px + 1))] = 2
    # Initialize variables
    pAll = np.ones_like(c[nz]) * 2
    Scales = np.ones_like(pAll)
    vAll = np.zeros_like(pAll)
    # 10 layers, Increasing the number of scale levels per octave can help in detecting features at finer scale intervals.
    s = 10
    # curr_filter = 1
    scales = {}
    print("Performing Gaussian Filter")
    for o in octave_values:
        scales[o] = {}
        sigma, t = compute_gaussian_params(o, 1, s)
        Gp = gaussian_filter(c, o, truncate=t, order=0)
        scales[o][1] = sigma
        # blur2
        sigma, t = compute_gaussian_params(o, 2, s)
        Gc = gaussian_filter(c, sigma, truncate=t, order=0)
        scales[o][2] = sigma
        Lp = Gp - Gc
        Gp = []
        # blur3
        sigma, t = compute_gaussian_params(o, 3, s)
        Gn = gaussian_filter(c, sigma, truncate=t, order=0)
        scales[o][3] = sigma
        # Lp = Gp - Gc
        Lc = Gc - Gn
        locMaxP = maximum_filter(Lp, footprint=np.ones((fp_size, fp_size)), mode='constant')
        locMaxC = maximum_filter(Lc, footprint=np.ones((fp_size, fp_size)), mode='constant')
        for i in range(3, s + 2):
            # curr_filter += 1
            Gc = Gn
            sigma, t = compute_gaussian_params(o, i, s)
            Gn = gaussian_filter(c, sigma, truncate=t, order=0)
            scales[o][i + 1] = sigma
            Ln = Gc - Gn
            dist_params = expon.fit(np.abs(Lc[nz]))
            pval = 1 - expon.cdf(np.abs(Lc[nz]), *dist_params)
            locMaxN = maximum_filter(Ln, footprint=np.ones((fp_size, fp_size)), mode='constant')
            willUpdate = np.logical_and.reduce((
                Lc[nz] > vAll,
                Lc[nz] == locMaxC[nz],
                np.logical_or(Lp[nz] == locMaxP[nz], Ln[nz] == locMaxN[nz]),
                Lc[nz] > locMaxP[nz],
                Lc[nz] > locMaxN[nz]
            ))
            vAll[willUpdate] = Lc[nz][willUpdate]
            Scales[willUpdate] = scales[o][i]
            pAll[willUpdate] = pval[willUpdate]
            Lp = Lc
            Lc = Ln
            locMaxP = locMaxC
            locMaxC = locMaxN
    pFound = pAll != 2
    if len(pFound) < 10000:
        return []
    print('Filter significant interactions')
    _, pCorrect, _, _ = multipletests(pAll[pFound], method='fdr_bh')
    pAll[pFound] = pCorrect
    print('Extract interaction')
    o = np.ones_like(c)
    o[nz] = pAll
    sig_count = np.sum(o < pt)  # change
    x, y = np.unravel_index(np.argsort(o.ravel()), o.shape)
    so = np.ones_like(c)
    so[nz] = Scales
    x = x[:sig_count]
    y = y[:sig_count]
    xyScales = so[x, y]
    nonsparse = x != 0
    for i in range(len(xyScales)):
        s = math.ceil(xyScales[i])
        c1 = np.sum(nz[x[i] - s:x[i] + s + 1, y[i] - s:y[i] + s + 1]) / ((2 * s + 1) ** 2)
        s = 2 * s
        c2 = np.sum(nz[x[i] - s:x[i] + s + 1, y[i] - s:y[i] + s + 1]) / ((2 * s + 1) ** 2)
        if c1 < st or c2 < 0.6:
            nonsparse[i] = False
    x = x[nonsparse]
    y = y[nonsparse]
    if len(x) == 0:
        return []
    def kth_diag_indices(a, k):
        rows, cols = np.diag_indices_from(a)
        if k < 0:
            return rows[-k:], cols[:k]
        elif k > 0:
            return rows[:-k], cols[k:]
        else:
            return rows, cols
    def nz_mean(vals):
        return np.mean(vals[vals != 0])
    def diag_mean(k, map):
        return nz_mean(map[kth_diag_indices(map, k)])
    print("Label interaction")
    means = np.vectorize(diag_mean, excluded=['map'])(k=y - x, map=c)
    passing_indices = c[x, y] > 2 * means  # change
    if len(passing_indices) == 0 or np.sum(passing_indices) == 0:
        return []
    x = x[passing_indices]
    y = y[passing_indices]
    label_matrix = np.zeros((np.max(y) + 2, np.max(y) + 2), dtype=np.float32)
    label_matrix[x, y] = o[x, y] + 1
    label_matrix[x + 1, y] = 2
    label_matrix[x + 1, y + 1] = 2
    label_matrix[x, y + 1] = 2
    label_matrix[x - 1, y] = 2
    label_matrix[x - 1, y - 1] = 2
    label_matrix[x, y - 1] = 2
    label_matrix[x + 1, y - 1] = 2
    label_matrix[x - 1, y + 1] = 2
    # Connected Components with 8-Connectivity
    num_features = scipy_measurements.label(
        label_matrix, output=label_matrix, structure=np.ones((3, 3)))
    # finalize out
    out = []
    for label in tqdm(range(1, num_features + 1)):
        indices = np.argwhere(label_matrix == label)
        i = np.argmin(o[indices[:, 0], indices[:, 1]])
        _x, _y = indices[i, 0], indices[i, 1]
        out.append([_x + start, _y + start, o[_x, _y], so[_x, _y]])
    return out

def process_block(i, start, end, overlap_size, cc, distance_in_px, octave_values, o, st, pt, ignore_diags, fp_size):
    print("Starting block ", i + 1, "/", len(start), "...", sep='')
    loops = SIFT(cc, start[i], end[i], distance_in_px=distance_in_px,
                     octave_values=octave_values, st=st, pt=pt,ignore_diags=ignore_diags, fp_size=fp_size)
    for loop in list(loops):
        if loop[0] >= start[i] + overlap_size[i] or loop[1] >= start[i] + overlap_size[i]:
            o.append([loop[0], loop[1], loop[2], loop[3]])
    print("Block", i + 1, "done.")

def regulator(clr, chrom, sigma0=1.6, pt=0.1, st=0.8, octaves=2, nprocesses=4, distance_filter=2000000,
              ignore_diags=2, fp_size=5, intensity_cut=0.2):
    octave_values = [sigma0 * (2 ** i) for i in range(octaves)]
    distance_in_bp = distance_filter
    midpoints = ((clr.bins().fetch(chrom)['start'] + clr.bins().fetch(chrom)['end'])/2).values
    print("Reading contact map...")
    x, y, v, start, end = read_cooler_varbins(clr, distance_in_bp, chrom,intensity_cut=intensity_cut)
    # no contact contained in map
    if len(x) == 0:
        return 0
    else:
        # overlap_size
        overlap_size = [-1] + list(np.array(end[:-1])-np.array(start[1:]))
        with Manager() as manager:
            o = manager.list()
            i = 0
            processes = []
            for i in range(len(start)):
                # extract the currnet block
                indx = np.logical_and.reduce((x >= start[i], x < end[i], y >= start[i], y < end[i]))
                xc = x[indx] - start[i]
                yc = y[indx] - start[i]
                vc = v[indx]
                chunksize = end[i] - start[i]
                cc = np.zeros((chunksize, chunksize))
                cc[xc, yc] = vc
                # find distance_in_px
                distance_in_px = find_distance_in_px(xc,yc,midpoints,distance_in_bp)
                print(f'distance in px: {distance_in_px}')
                p = Process(target=process_block, args=(
                    i, start, end, overlap_size, cc, distance_in_px, octave_values, o, st, pt, ignore_diags,fp_size))
                p.start()
                processes.append(p)
                if len(processes) >= nprocesses or i == (len(start) - 1):
                    for p in processes:
                        p.join()
                    processes = []
            # o_corrected = [[e[0],e[1],e[2]/pval_weights[e[1]-e[0]],e[3]] for e in list(o)]
            return list(o)

