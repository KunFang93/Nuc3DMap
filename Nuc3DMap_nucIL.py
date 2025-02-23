import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, triu
from numba import njit
import cooler
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from Nuc3DMap_SIFT import regulator
from natsort import natsort_keygen
from sklearn.linear_model import LinearRegression
from Nuc3DMap_utilities import GenomicAnnot
pd.options.mode.chained_assignment = None

@njit
def bincount_mod2(out, rows, r, C, V):
    N = len(V)
    for i in range(N):
        out[rows[r[i]], rows[C[i]]] += V[i]
    return out

def blockSum(X, rows):
    nrows = rows.max() + 1
    r, C = X.nonzero()
    V = X[r, C].A1
    out = np.zeros((nrows, nrows))
    final_out = bincount_mod2(out, rows, r, C, V)
    final_coo = coo_matrix(final_out)
    return final_coo

def calSparsity(A):
    sparsity = 1.0 - (np.count_nonzero(A) / float(A.size))
    return sparsity

def sizeScale_blocksum(matrix_coo, size):
    print("Size scaling")
    # Extract row, col, and data from the COO matrix
    matrix_coo_up = triu(matrix_coo)
    upper_rows, upper_cols, upper_val = matrix_coo_up.row, matrix_coo_up.col, matrix_coo_up.data
    # Scale the values using the size array for the upper triangle
    scaled_values_upper = [v / (size[r] * size[c]) for r, c, v in zip(upper_rows, upper_cols, upper_val)]
    # Mirror the scaled values to the lower triangle
    all_rows_coo = list(upper_rows) + list(upper_cols)
    all_cols_coo = list(upper_cols) + list(upper_rows)
    all_values_coo = scaled_values_upper + scaled_values_upper
    # convert to coo
    new_coo = coo_matrix((all_values_coo, (all_rows_coo, all_cols_coo)), shape=matrix_coo.shape)
    # double count diagonal
    final_coo = new_coo - eye(new_coo.shape[0]).multiply(new_coo.diagonal() / 2)
    # Construct and return the scaled symmetric COO matrix
    return final_coo.tocoo()

def fill_array_from_df(arr, start, end, value):
    arr[start:end + 1] = value

@njit
def calDistance(rows, cols, midpoints):
    distances = []
    for row, col in zip(rows, cols):
        distances.append(np.abs(midpoints[row] - midpoints[col]))
    return distances

@njit
def _makeEqualSizeBins(rows, cols, data, distances, numBins):
    # Calculate in-range observation's summation
    observationInRange = np.sum(data)
    observationsPerBin = observationInRange / numBins
    # Initialize arrays
    current_dist = np.empty(len(rows), dtype=np.float64)
    bin_idx = 0
    bin_last_idx = 0  # Added this to remember the start of the current bin
    current_count_sum = 0.0
    current_bin_sum = 0.0
    current_dist_sum = 0.0
    # Separate arrays for binStats components
    binStats_counts = np.zeros(numBins)
    binStats_contact = np.zeros(numBins)
    binStats_distances = np.zeros(numBins)
    binStats_range = np.empty((numBins, 2), dtype=np.float64)
    bintracker = 0
    # Iterate through the sorted pixels to fill the bins
    for i in range(len(rows)):
        # don't count row==col
        row, col, val, dist = rows[i], cols[i], data[i], distances[i]
        current_dist[bin_idx] = dist
        bin_idx += 1
        current_bin_sum += val
        current_dist_sum += dist
        current_count_sum += 1
        # Check if the current bin has reached the desired observations
        if current_bin_sum >= observationsPerBin:
            binStats_counts[bintracker] = current_count_sum
            binStats_contact[bintracker] = current_bin_sum
            binStats_distances[bintracker] = current_dist_sum
            binStats_range[bintracker, 0] = current_dist[bin_last_idx:bin_idx].min()
            binStats_range[bintracker, 1] = current_dist[bin_last_idx:bin_idx].max()
            # Reset arrays and sums
            bin_last_idx = bin_idx
            current_bin_sum = 0.0
            current_dist_sum = 0.0
            current_count_sum = 0.0
            bintracker += 1
    # Fill the last bin
    if bin_idx > bin_last_idx:
        binStats_counts[bintracker] = current_count_sum
        binStats_contact[bintracker] = current_bin_sum
        binStats_distances[bintracker] = current_dist_sum
        binStats_range[bintracker, 0] = current_dist[bin_last_idx:bin_idx].min()
        binStats_range[bintracker, 1] = current_dist[bin_last_idx:bin_idx].max()
    return binStats_counts, binStats_contact, binStats_distances, binStats_range

@njit
def countAllPossiblePair(sorted_data, bins):
    bin_counts = np.zeros(len(bins) - 1, dtype=np.int64)
    for bin_idx in range(len(bins) - 1):
        i = 0
        j = 1
        while i < len(sorted_data) and j < len(sorted_data):
            diff = sorted_data[j] - sorted_data[i]
            if diff < bins[bin_idx]:
                j += 1
            elif bins[bin_idx] <= diff < bins[bin_idx + 1]:
                bin_counts[bin_idx] += j - i
                i += 1
            else:
                i += 1
                j = i + 1
    return bin_counts


@njit
def DistBalance(data, distBinidx, DistBias):
    data_bal = np.zeros(len(data))
    for i in range(len(data)):
        val, curBinidx = data[i], distBinidx[i]
        curBias = DistBias[curBinidx]
        curbal = val * curBias
        data_bal[i] = curbal
    return data_bal

class DistanceBalance(object):
    def __init__(self, coo_mat_KR_chr, bins_chr, numBins=200, plot_mark=False):
        self.coo_mat_KRnormed = coo_mat_KR_chr
        self.bin_chr = bins_chr
        self.numBins = numBins
        self.plot_mark = plot_mark

    def _sortbyDistance(self, rows, cols, data, distances):
        # Create a sorted index for the COO matrix based on distances
        sorted_index = np.argsort(distances)
        # Sort the rows, cols, and data of the COO matrix using the sorted index
        distances_srt = distances[sorted_index]
        rows_srt = rows[sorted_index]
        cols_srt = cols[sorted_index]
        data_srt = data[sorted_index]
        return rows_srt, cols_srt, data_srt, distances_srt

    def makeEqualSizeBins(self, rows, cols, data, distances, numBins):
        counts, contact, distances, ranges = _makeEqualSizeBins(rows, cols, data, distances, numBins)
        # 0: no. of possible pairs w/in this range of distances
        # 1: sumoverallContactCounts in bin
        # 2: Sumoveralldistances in this bin in distScaling vals
        # 3: range of distances in this bin
        # Convert arrays to desired list format
        binStats = pd.DataFrame({
            'numContactCounts': counts,
            'SumContactCounts': contact,
            'SumDistances': distances,
            'binStart': [ranges[i, 0] for i in range(numBins)],
            'binEnd': [ranges[i, 1] for i in range(numBins)]
        })
        return binStats

    def calDistBias(self, avgCC, minBias=0.01, maxBias=1):
        # maximum residual allowed for spline is set to min(y)^2
        splineError = min(avgCC) * min(avgCC)
        # fit with spline
        ius = UnivariateSpline(np.arange(len(avgCC)), avgCC, s=splineError)
        # generate fitted mean
        fit = ius(np.arange(len(avgCC)))
        return fit

    def distbalance(self):
        print("Performing distance balancing")
        # distance balanced
        midpoints = ((self.bin_chr['start'] + self.bin_chr['end']) / 2).values
        distances = calDistance(self.coo_mat_KRnormed.row, self.coo_mat_KRnormed.col, midpoints)
        rows_srt, cols_srt, data_srt, distances_srt = self._sortbyDistance(self.coo_mat_KRnormed.row,
                                                                           self.coo_mat_KRnormed.col,
                                                                           self.coo_mat_KRnormed.data,
                                                                           np.array(distances))
        binStats = self.makeEqualSizeBins(rows_srt, cols_srt, data_srt, distances_srt, self.numBins)
        # aggregate if same bins
        binStats = binStats.groupby(['binStart']).agg({
            'numContactCounts': np.sum,
            'SumContactCounts': np.sum,
            'SumDistances': np.sum,
            'binEnd': np.max
        }).reset_index()
        # empirical scaling factor is needed if use all possible pairs (otherwise the CC is too small as the resolution is too small and depth might not sufficient),
        # so use this simple version; todo: use all possible pairs
        binStats['AvgCC'] = binStats['SumContactCounts'] / binStats['numContactCounts']
        binStats['AvgDist'] = binStats['SumDistances'] / binStats['numContactCounts']
        if self.plot_mark:
            # fill binStats[bins][0] with all possible size
            binStats['AllPossPair'] = countAllPossiblePair(midpoints,
                                                           np.array([0] + binStats['binEnd'].values.tolist()))
            binStats.loc[binStats['binStart'] == 0, 'AllPossPair'] += len(self.bin_chr)
            binStats['AvgCC_allposs'] = binStats['SumContactCounts'] / binStats['AllPossPair']
            # pseduocount for first bin
            binStats.loc[binStats['binStart'] == 0, 'AvgCC_allposs'] = binStats['AvgCC_allposs'].values[1] + \
                                                                       binStats.loc[
                                                                           binStats['binStart'] == 0, 'AvgCC_allposs']
            # visualize
            fig, ax = plt.subplots(1, 1)
            ax.plot(np.arange(len(binStats) - 1), binStats['AvgCC_allposs'].values[1:])
            # Select 10 evenly spaced indices for xticks
            selected_indices = np.linspace(0, len(binStats) - 2, 10, dtype=int)
            ax.set_xticks(selected_indices + 0.5)  # Positioning at the center of the cell
            ax.set_xticklabels(binStats['AvgDist'].round(2).values[1:][selected_indices], rotation=45)
            plt.tight_layout()
            plt.savefig(f'/Distance-CC_relationship.png')
            plt.close()
        # empirically consider first bins' Bias is 1, So calculate distance bias from second bins
        distBias = 1 / self.calDistBias(binStats['AvgCC'])
        # distBias = 1 / binStats['AvgCC'].values
        binStats['DistBias'] = list(distBias)
        distBinidx = np.searchsorted(np.concatenate(([0], binStats['binEnd'])), distances_srt, side='right') - 1
        data_balanced = DistBalance(data_srt, distBinidx, binStats['DistBias'].values)
        coo_mat_KRnormed_DistBal = coo_matrix((data_balanced, (rows_srt, cols_srt)),
                                              shape=(len(self.bin_chr), len(self.bin_chr)))
        return coo_mat_KRnormed_DistBal


def getpixels(coo_mat_triu, relevel_binid, relevel_dfid):
    pixels_df = pd.DataFrame({
        'bin1_id': coo_mat_triu.row + relevel_binid,
        'bin2_id': coo_mat_triu.col + relevel_binid,
        'count': coo_mat_triu.data
    })
    pixels_df.index = relevel_dfid + pixels_df.index.values
    return pixels_df


def chrom_iterator(coo_list, bins_list, chroms):
    relevel_df_idx = 0
    for idx, chr in enumerate(chroms):
        print(chr)
        coo_mat = coo_list[idx]
        coo_mat_triu = triu(coo_mat)
        relevel_binid = bins_list[idx]['bin_id'].min()
        yield getpixels(coo_mat_triu, relevel_binid, relevel_df_idx)
        relevel_df_idx += len(coo_mat_triu.data)


def save_cooler(cooler_path, bins_list, coo_list, chroms):
    chrom_iter = chrom_iterator(coo_list, bins_list, chroms)
    bins_df = pd.concat(bins_list)
    cooler.create_cooler(cool_uri=cooler_path,
                         bins=bins_df,
                         pixels=chrom_iter,
                         ordered=True,
                         dtypes={'count': np.float32})
    return None

def count_significant_decimal_places(number):
    # Convert the number to a string
    number_str = str(number)

    # Find the decimal point and start counting after it
    decimal_index = number_str.find('.')
    count = 1
    # Count the number of significant digits after the decimal point
    for digit in number_str[decimal_index + 1:]:
        if digit == '0':
            count += 1
        else:
            break
    return count

def extEnd(blocked_clr, IL_df, mode=None):
    # mode constant/adjacent
    if mode == 'constant':
        extsize = 3000
        loc1_mid = IL_df[['start1', 'end1']].mean(axis=1).astype(int)
        loc2_mid = IL_df[['start2', 'end2']].mean(axis=1).astype(int)
        IL_df['start1'] = loc1_mid - extsize
        IL_df['end1'] = loc1_mid + extsize
        IL_df['start2'] = loc2_mid - extsize
        IL_df['end2'] = loc2_mid + extsize
        IL_extend_df = IL_df[['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'sidx', 'eidx', 'FDR', 'Intensities']]
    elif mode == 'adjacent':
        IL_extend_list = []
        chroms = blocked_clr.chromnames
        for chrom in chroms:
            cur_bins = blocked_clr.bins().fetch(chrom)
            cur_il = IL_df[IL_df['chrom1'] == chrom]
            start_idmap = dict(zip(cur_bins['arrayid'], cur_bins['start']))
            end_idmap = dict(zip(cur_bins['arrayid'], cur_bins['end']))
            cur_il['start1_ext'] = (cur_il['sidx'] - 1).map(start_idmap)
            cur_il['end1_ext'] = (cur_il['sidx'] + 1).map(end_idmap)
            cur_il['start2_ext'] = (cur_il['eidx'] - 1).map(start_idmap)
            cur_il['end2_ext'] = (cur_il['eidx'] + 1).map(end_idmap)
            IL_extend_list.append(cur_il)
        IL_extend_df = pd.concat(IL_extend_list)
        IL_extend_df = IL_extend_df[
            ['chrom1', 'start1_ext', 'end1_ext', 'chrom2', 'start2_ext', 'end2_ext', 'sidx', 'eidx', 'FDR',
             'Intensities']]
        IL_extend_df.columns = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'sidx', 'eidx', 'FDR',
                                'Intensities']
    else:
        IL_extend_df = IL_df.copy()
    return IL_extend_df

def build_RGMap(imhic_clr, tads_df, outcool):
    tads_df['bin_id'] = np.arange(len(tads_df))
    tads_df_gp = tads_df.groupby('chrom')
    blockscale_coos = []
    blockscale_bins = []
    tag2id = {'gap': 0, 'boundary': 1, 'domain': 2}
    chroms = imhic_clr.chromnames
    for chrom in chroms:
        print(f"Processing {chrom}")
        bins_chr = imhic_clr.bins().fetch(chrom)
        # based on balanced matrix
        coo_mat_kr = imhic_clr.matrix(sparse=True).fetch(chrom)
        print("Distance Normalization")
        # apply distance balance to csr_mat
        distanceBalance = DistanceBalance(coo_mat_kr, bins_chr)
        coo_mat_krdnormed = distanceBalance.distbalance()
        csr_mat = coo_mat_krdnormed.tocsr()
        cur_tads_df = tads_df_gp.get_group(chrom)
        cur_tads_df['arrayid'] = np.arange(len(cur_tads_df))
        # Create a numpy array based with same size of csr_mat
        rows_array_tadid = np.zeros(csr_mat.shape[0], dtype=int)
        cur_tads_df.apply(
            lambda row: fill_array_from_df(rows_array_tadid, row['snucidx'], row['enucidx'], row['arrayid']),
            axis=1)
        # Use the sparse_matrix_mult_sparseX_mod2 function to compute the intermediate matrix for the loaded data
        print("Performing blocked-wise summation")
        blocksum_coo = blockSum(csr_mat, rows_array_tadid)
        # normalize by region size
        print("Normalized by block size")
        cur_tads_df['size'] = cur_tads_df['enucidx'] - cur_tads_df['snucidx'] + 1
        blocksum_coo_scale = sizeScale_blocksum(blocksum_coo, cur_tads_df['size'].values)
        # save to list
        blockscale_coos.append(blocksum_coo_scale)
        cur_tads_df['tag'] = cur_tads_df['tag'].map(tag2id)
        blockscale_bins.append(cur_tads_df)

    print("Writing to .cool")
    # save blocked_scaled matrix to new cooler file
    save_cooler(outcool, blockscale_bins, blockscale_coos, chroms)
    return None

@njit
def pixelInDistanceInetensity(rows, cols, data, midpoints, dist_lowcut, dist_upcut, qcut):
    rows_filt, cols_filt, data_filt = [], [], []
    for row, col, val in zip(rows, cols, data):
        cur_dist = np.abs(midpoints[row] - midpoints[col])
        if cur_dist <= dist_upcut and cur_dist >= dist_lowcut:
            rows_filt.append(row)
            cols_filt.append(col)
            data_filt.append(val)
        else:
            continue
    # get quantila cut
    qcut_val = np.quantile(data_filt, qcut)
    # further filter low intensity pixel
    rows_qfilt, cols_qfilt, data_qfilt = [], [], []
    for row, col, val in zip(rows_filt, cols_filt, data_filt):
        if val >= qcut_val:
            rows_qfilt.append(row)
            cols_qfilt.append(col)
            data_qfilt.append(val)
        else:
            continue
    return rows_qfilt, cols_qfilt, data_qfilt

def pixelGroup(rows, cols, data, winsize):
    max_group_size = winsize ** 2 * 2
    grouped = []
    current_group = [(rows[0], cols[0], data[0])]
    for row, col, val in zip(rows[1:], cols[1:], data[1:]):
        # Check if the current pixel is close to the last pixel in the current group
        if abs(row - current_group[-1][0]) <= winsize and abs(col - current_group[-1][1]) <= winsize:
            current_group.append((row, col, val))
        else:
            # Process the current group before starting a new one
            if len(current_group) > max_group_size:
                # Subdivide the group
                for i in range(0, len(current_group), max_group_size):
                    subgroup = current_group[i:i + max_group_size]
                    grouped.append(subgroup)
            else:
                grouped.append(current_group)
            current_group = [(row, col, val)]
    # Handle the last group
    if current_group:  # If there's anything left in the current group
        if len(current_group) > max_group_size:
            for i in range(0, len(current_group), max_group_size):
                subgroup = current_group[i:i + max_group_size]
                grouped.append(subgroup)
        else:
            grouped.append(current_group)
    return grouped

def pixelFinalize(grouped):
    rows_final, cols_final, data_final = [], [], []
    for group in grouped:
        # Select the pixel with the highest value from each group
        highest_pixel = max(group, key=lambda x: x[2])
        rows_final.append(highest_pixel[0])
        cols_final.append(highest_pixel[1])
        data_final.append(highest_pixel[2])
    return rows_final, cols_final, data_final

def LMDistMakeUp(rows, cols, data, midpoints, dist_lowcut, dist_upcut, qcut=0.999, winsize=5):
    # Finding high intensity pixels with qcut
    rows_qfilt, cols_qfilt, data_qfilt = pixelInDistanceInetensity(rows, cols, data, midpoints, dist_lowcut, dist_upcut, qcut)
    # Grouping adjacent pixels
    grouped = pixelGroup(rows_qfilt, cols_qfilt, data_qfilt, winsize)
    # Finalize
    rows_final, cols_final, data_final = pixelFinalize(grouped)
    return rows_final, cols_final, data_final

def QuantileDetection(rg_clr,dist_lowcut=100000,dist_upcut=2000000,qcut=0.999):
    print("Quantile Detection")
    il_final_list = []
    chroms = rg_clr.chromnames
    for chrom in chroms:
        cur_bins = rg_clr.bins().fetch(chrom).copy()
        if len(cur_bins) <= 10:
            print(f"{chrom} has too less block to detect, skip")
            continue
        cur_mat = rg_clr.matrix(balance=False, sparse=True).fetch(chrom)
        midpoints = ((cur_bins['start'] + cur_bins['end']) / 2).values
        try:
            rows_final, cols_final, data_final = LMDistMakeUp(cur_mat.row, cur_mat.col, cur_mat.data, midpoints,
                                                              dist_lowcut=dist_lowcut, dist_upcut=dist_upcut,qcut=qcut)
        except IndexError:
            print(f"Not pixels met requirement, {chrom} Skipped")
            continue
        # form bedpe like dataframe
        cur_bins.reset_index(drop=True, inplace=True)
        rows_final_df = cur_bins.loc[rows_final, ['chrom', 'start', 'end', 'arrayid']].reset_index(drop=True)
        rows_final_df.columns = ['chrom1', 'start1', 'end1', 'sidx']
        cols_final_df = cur_bins.loc[cols_final, ['chrom', 'start', 'end', 'arrayid']].reset_index(drop=True)
        cols_final_df.columns = ['chrom2', 'start2', 'end2', 'eidx']
        cur_il_final = pd.concat([rows_final_df, cols_final_df], axis=1)
        cur_il_final['Intensities'] = data_final
        # not keep upper mat
        cur_il_final = cur_il_final[cur_il_final['sidx'] < cur_il_final['eidx']]
        il_final_list.append(cur_il_final)
    il_final_df = pd.concat(il_final_list)
    il_final_df['chrom1'] = il_final_df['chrom1'].cat.remove_unused_categories()
    return il_final_df

def Q_FDR_DS(q_df,sift_df):
    q_chroms = []
    # add FDR and Detection scale
    sift_df_gp = sift_df.groupby('chrom1')
    for chrom, chrom_df in q_df.groupby('chrom1'):
        try:
            cur_sift = sift_df_gp.get_group(chrom)
            min_I = chrom_df['Intensities'].min()
            cur_sift_filt = cur_sift[cur_sift['Intensities'] >= min_I]
            # Create a linear regression model
            fdr_model = LinearRegression()
            # Fit the model
            fdr_model.fit(cur_sift_filt['Intensities'].values.reshape(-1, 1), cur_sift_filt['FDR'].values)
            chrom_fdr = fdr_model.predict(chrom_df['Intensities'].values.reshape(-1, 1))
            ds_model = LinearRegression()
            ds_model.fit(cur_sift_filt['Intensities'].values.reshape(-1, 1), cur_sift_filt['DETECTION SCALE'].values)
            chrom_ds = ds_model.predict(chrom_df['Intensities'].values.reshape(-1, 1))
            chrom_fdr[chrom_fdr <= 0] = 0.00000
            chrom_df['FDR'] = np.round(chrom_fdr, 5)
            chrom_df['DETECTION SCALE'] = np.round(chrom_ds, 5)
            # make sure all positive
            q_chroms.append(chrom_df)
        except KeyError:
            # use the constant FDR and DETECTION SCALE
            chrom_df['FDR'] = 0.00001
            chrom_df['DETECTION SCALE'] = 0.6155
            q_chroms.append(chrom_df)
    q_final = pd.concat(q_chroms)
    # reorder columns
    q_final = q_final[sift_df.columns]
    return q_final
def NucIL_detection(blocksum_clr, distance_up_filter = 2000000, distance_low_filter = 5000, pt = 0.1, sigma0 = 0.5,
                    intensity_cut = 0.0, nprocesser = 20, extmode = None):
    chroms = blocksum_clr.chromnames
    ignore_diags = 2
    sts = [0.25, 0.75]  # spatial threshold, used to filter out interactions based on spatial density. default 0.5
    octaves = 2  # Increasing the number of octaves means that the algorithm will search for key points across a larger range of scales.
    fp_size = 5
    st_nucloop = []
    for st in sts:
        nucloop_list = []
        for chrom in chroms:
            print(f"Processing {chrom}")
            bins_df = blocksum_clr.bins().fetch(chrom).reset_index(drop=True)
            o = regulator(blocksum_clr, chrom, sigma0=sigma0, pt=pt, st=st, octaves=octaves,
                          nprocesses=nprocesser, distance_filter=distance_up_filter, ignore_diags=ignore_diags,
                          fp_size=fp_size, intensity_cut=intensity_cut)
            if o == 0:
                continue
            else:
                o_df = pd.DataFrame({
                    'chrom1': [chrom] * len(o),
                    'start1': bins_df.loc[np.array([l[0] for l in o]), 'start'].values,
                    'end1': bins_df.loc[np.array([l[0] for l in o]), 'end'].values,
                    'sidx': [l[0] for l in o],
                    'chrom2': [chrom] * len(o),
                    'start2': bins_df.loc[np.array([l[1] for l in o]), 'start'].values,
                    'end2': bins_df.loc[np.array([l[1] for l in o]), 'end'].values,
                    'eidx': [l[1] for l in o],
                    'FDR': [l[2] for l in o],
                    'DETECTION SCALE': [l[3] for l in o],
                })
                # in case o_df is empty
                if len(o_df) == 0:
                    print(
                        f"There is no interaction locus identified from {chrom}, consider use smaller st or sigma0 params")
                    continue
                else:
                    # assign intensity col
                    mat = blocksum_clr.matrix(balance=False, sparse=True).fetch(chrom)
                    mat_csr = mat.tocsr()
                    intensities = mat_csr[o_df['sidx'], o_df['eidx']].tolist()[0]
                    o_df['Intensities'] = intensities
                    # make sure the loops are in range
                    o_df = o_df[
                        np.abs(o_df[['start2', 'end2']].mean(axis=1) - o_df[['start1', 'end1']].mean(
                            axis=1)) >= distance_low_filter]
                    o_df = o_df[
                        np.abs(o_df[['start2', 'end2']].mean(axis=1) - o_df[['start1', 'end1']].mean(
                            axis=1)) <= distance_up_filter]
                    nucloop_list.append(o_df)
        # write
        nucloops_df = pd.concat(nucloop_list)
        # sorted df
        nucloops_df = nucloops_df.sort_values(by=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'],
                                              key=natsort_keygen())
        st_nucloop.append(nucloops_df)
    # NucIL from SIFT
    nucloops_df_sift = pd.concat(st_nucloop).drop_duplicates()
    print("SIFT Detection Finished")
    # In case some high intensity pixels are missed because the background comparison, sepcifically in middle ranges
    nucloops_df_qcut = QuantileDetection(blocksum_clr,dist_lowcut=100000, dist_upcut=2000000, qcut=0.999)
    # Adjust format by comparative analysis
    nucloops_df_qcut_final = Q_FDR_DS(nucloops_df_qcut,nucloops_df_sift)
    # combine all NucIL
    nucloops_df_final = pd.concat([nucloops_df_sift,nucloops_df_qcut_final])
    # remove duplicated row
    nucloops_df_final.drop_duplicates(['sidx','eidx'],inplace=True)
    print("Quantile Detection Finished")
    # extend end
    nucloops_df_final = nucloops_df_final.sort_values(by=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'],
                                                      key=natsort_keygen())

    nucloops_df_final_ext = extEnd(blocksum_clr, nucloops_df_final, mode=extmode)
    # generate bedpe for IGV
    nucloops_bedpe = nucloops_df_final_ext[
        ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'Intensities']]
    nucloops_bedpe['Intensities'] = (
            (10 ** count_significant_decimal_places(nucloops_bedpe['Intensities'].min())) * nucloops_bedpe[
        'Intensities']).round(
        0).astype(int)
    return nucloops_df_final_ext, nucloops_bedpe

def transfer_NucLoop(nucil_df, geneannot):
    """
    Transfer NucIL to P(romoter)-related NucLoops
    :param nucil_df:
    :param geneannot:
    :return:
    """
    def normalize_annotation(annot):
        """Normalize annotation to a single character code."""
        mapping = {'Distal': 'D1', 'Promoter': 'P1', 'Genebody': 'G1'}
        return mapping.get(annot, annot)

    def find_common_elements(detail1, detail2):
        """Find common elements between detail1 and detail2, ignoring _bi suffix."""
        set1 = set(detail1.replace("_bi", "").split(","))
        set2 = set(detail2.replace("_bi", "").split(","))
        return set1.intersection(set2)

    def assign_annotation(row):
        """Correctly assign annotation based on updated rules, oversight correction, and common genes."""
        # Normalize annotations
        annot1 = normalize_annotation(row['annot1'])
        annot2 = normalize_annotation(row['annot2'])

        # Initialize the result variable for the annotation
        result_annotation = 'unknown'

        # Rule 1: Check for common elements (Adjust this logic if necessary for finding common genes)
        common_elements = find_common_elements(row['detail1'], row['detail2'])
        common_genes = list(common_elements)
        if common_elements:
            omitorient = {'P1D1': 'D1P1', 'G1P1': 'P1G1', 'G1D1': 'D1G1'}
            combination = ''.join(sorted([annot1, annot2], key=lambda x: x[::-1]))
            if combination in ['D1P1', 'P1G1', 'D1G1', 'D1D1', 'G1G1']:
                result_annotation = combination
            elif combination in list(omitorient.keys()):
                result_annotation = omitorient[combination]
        # Rule 2: Both are Promoters
        elif annot1 == 'P1' and annot2 == 'P1':
            result_annotation = 'P2P1'
            common_genes.append(row['detail1'])
            common_genes.append(row['detail2'])
        # Rule 3: if one end is Promoter, then unknow should be D1P1
        elif annot1 == 'P1' or annot2 == 'P1':
            result_annotation = 'D1P1'
            if annot1 == 'P1':
                common_genes.append(row['detail1'])
            else:
                common_genes.append(row['detail2'])
        # Rule 4: Both are Genebody not same gene
        elif annot1 == 'G1' and annot2 == 'G1':
            result_annotation = 'G2G1'
            common_genes.append(row['detail1'])
            common_genes.append(row['detail2'])
        # Rule 5: one in genebody but another in distal of another gene
        elif (annot1 == 'G1' and annot2 == 'D1') or (annot1 == 'D1' and annot2 == 'G1'):
            result_annotation = 'D2G1'
            if annot1 == 'G1':
                common_genes.append(row['detail1'])
            else:
                common_genes.append(row['detail2'])
        # Rule 6: Both Distal not same gene
        elif annot1 == 'D1' and annot2 == 'D1':
            result_annotation = 'D2D1'
            common_genes.append('Undefined')
        # Rule 7: Desert related loop
        elif annot1 == 'Desert' or annot2 == 'Desert':
            result_annotation = 'Desert'
            common_genes.append('Undefined')
        # Return both the annotation and the common genes
        else:
            print(f"Annot1:{annot1};Annot2{annot2}")
        return result_annotation, ','.join(list(set(common_genes)))

    print("Annotating NucIL ends")
    genes_df = pd.read_table(geneannot, header=None, names=['Chromosome', 'Start', 'End', 'Strand', 'GeneName', 'Count'])
    IL_df = nucil_df.copy()
    IL_df['ILID'] = np.arange(len(IL_df))
    # get different genomic region
    genomic_regions_ranges = {
        # prom:-1000 ~ 1000 of TTS
        'Promoter': [-5000, -1000, 1000, 5000],
        # 1000 of TTS to -1000 of tts
        'Genebody': [1000, 0],
        'Distal': [-250000, 5000, -5000, 250000]
    }
    IL_loc1 = IL_df[['chrom1', 'start1', 'end1', 'ILID']]
    IL_loc2 = IL_df[['chrom2', 'start2', 'end2', 'ILID']]

    IL_loc1.columns = ['Chromosome', 'Start', 'End', 'ILID']
    IL_loc2.columns = ['Chromosome', 'Start', 'End', 'ILID']
    print("Process Loc1")
    IL_loc1_geneannot = GenomicAnnot(IL_loc1, genes_df, genomic_region_ranges=genomic_regions_ranges, bidirect=True)
    IL_loc1_annot = IL_loc1_geneannot.annot()
    print("Process Loc2")
    IL_loc2_geneannot = GenomicAnnot(IL_loc2, genes_df, genomic_region_ranges=genomic_regions_ranges, bidirect=True)
    IL_loc2_annot = IL_loc2_geneannot.annot()

    IL_combined_annot = pd.concat([IL_loc1_annot, IL_loc2_annot], axis=1)
    IL_combined_annot.columns = ['chrom1', 'start1', 'end1', 'annot1', 'detail1',
                                 'chrom2', 'start2', 'end2', 'annot2', 'detail2']

    # Example of how to apply this function and create a new column with the results
    # Assuming IL_combined_annot is your DataFrame
    print("Screening P-related NucLoop")
    IL_combined_annot[['assigned_annotation', 'genes']] = IL_combined_annot.apply(lambda row: assign_annotation(row),
                                                                                  axis=1, result_type='expand')
    # targeted P1
    IL_combined_annot = IL_combined_annot[IL_combined_annot['assigned_annotation'].str.contains('P1')]
    return IL_combined_annot