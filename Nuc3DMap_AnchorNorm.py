import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from Nuc3DMap_smoothMD import MD_smoothing
from tqdm import tqdm

def MDplot(md_df, dcol='D', outdir='./', resolution=250, neighbours=200, plot_loess=True,
           plot_scatter=False, plot_outliers=True, outlier_n=6):
    samples = list(md_df.columns)[4:]
    combs = [comb for comb in itertools.combinations(samples, 2)]
    fignrow = np.max([len(combs) // 3, 1])
    if fignrow == 1:
        figncol = len(combs) % 3
    else:
        figncol = 3
    fig, axs = plt.subplots(fignrow, figncol, figsize=(15, 5))
    for idx, comb in enumerate(combs):
        M = np.log2((md_df[comb[0]] + 1) / (md_df[comb[1]] + 1))
        if fignrow != 1:
            ridx = idx // 3
            cidx = idx % 3
            MD_smoothing(M=M, D=md_df[dcol], title=f'{comb[0]}.vs.{comb[1]}', resolution=resolution,
                         neighbours=neighbours, xlab=dcol,
                         plot_loess=plot_loess, plot_scatter=plot_scatter, plot_outliers=plot_outliers,
                         outlier_n=outlier_n,
                         D_range=1, ax=axs[ridx][cidx])
        elif figncol != 1:
            MD_smoothing(M=M, D=md_df[dcol], title=f'{comb[0]}.vs.{comb[1]}', resolution=resolution,
                         neighbours=neighbours, xlab=dcol,
                         plot_loess=plot_loess, plot_scatter=plot_scatter, plot_outliers=plot_outliers,
                         outlier_n=outlier_n,
                         D_range=1, ax=axs[idx])
        else:
            MD_smoothing(M=M, D=md_df[dcol], title=f'{comb[0]}.vs.{comb[1]}', resolution=resolution,
                         neighbours=neighbours, xlab=dcol,
                         plot_loess=plot_loess, plot_scatter=plot_scatter, plot_outliers=plot_outliers,
                         outlier_n=outlier_n,
                         D_range=1, ax=axs)
    plt.tight_layout()
    plt.savefig(outdir, dpi=300)
    plt.close()

class AnchorNorm(object):
    def __init__(self, current_md, iterations=3, delta=0.01, frac=0.6):
        """
        AnchorNorm is for Normalizing margin value between two contact matrix
        :param current_md:
        :param iterations:
        :param delta:
        :param frac:
        :param thresh:
        """
        self.current_md = current_md
        self.iterations = iterations
        self.delta = delta
        self.frac = frac
    def _is_outlier(self, points, thresh=3.5):
        """
        copied from stackflow solution
        Returns a boolean array with True if points are outliers and False
        otherwise.

        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as outliers.

        Returns:
        --------
            mask : A numobservations-length boolean array.

        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
        """
        # Ensure points is at least 2D
        if len(points.shape) == 1:
            points = points[:, None]
        # Filter out zero elements (assuming you want to ignore zeros in any dimension)
        non_zero_points = points[np.all(points != 0, axis=-1)]
        # Proceed with your outlier detection on non-zero points
        if non_zero_points.size == 0:  # Check if all points were zeros
            return np.array([], dtype=bool)  # Or handle this case as you see fit
        median = np.median(non_zero_points, axis=0)
        diff = np.sum((non_zero_points - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        # Handle case where med_abs_deviation is zero to avoid division by zero
        if med_abs_deviation == 0:
            return np.full(points.shape[0], False)  # Assuming you want to mark all as non-outliers
        modified_z_score = 0.6745 * diff / med_abs_deviation
        # Generate mask for original points array
        # First, create a mask of False for all points
        mask = np.full(points.shape[0], False)
        # Find indices of non-zero points in the original array
        non_zero_indices = np.where(np.all(points != 0, axis=-1))[0]
        # Update mask with outlier detection result
        mask[non_zero_indices] = modified_z_score > thresh
        return mask
    def _lowessnorm(self, tab):
        """
        implement lowess method to normalize hic interaction frequences (IFs), borrowed idea from HiCcompare
        :param tab: dataframe contain at least 5 cols: chrom/start/end/D/IF1
        :param use_anchors: use digestion anchor information to perform Lowess
        :param iterations: the number of iterations for lowess weighting
        :param delta: Distance within which to use linear-interpolation instead of weighted regression
        :param frac: Between 0 and 1. The fraction of the data used when estimating each y-value.
        :return: lowess normalized IFs
        """
        # todo implement parallel processing
        # todo: plot lowess, confidence interval
        # todo: implement auto detection for frac using generalized cross validation
        # todo: implement fast loess for predction format if necessary
        # make matrix of IFs
        IF_mat = tab.iloc[:, 4:].values
        # make indicator matrix
        idx_mat = np.where(IF_mat != 0, 1, 0)
        # log the IF matrix
        IF_mat = np.log2(IF_mat + 1)
        n = IF_mat.shape[1]
        # calculate rowmeans (A)
        A = np.nanmean(IF_mat, axis=1)
        for j in range(n):
            M = IF_mat[:, j] - A
            # fit loess curve
            f = sm.nonparametric.lowess(exog=A, endog=M, frac=self.frac, it=self.iterations, delta=self.delta, return_sorted=False)
            # adjust
            IF_mat[:, j] = IF_mat[:, j] - f
        # anti-log the IF_mat
        IF_mat = (2 ** IF_mat) - 1
        # deal with negative values after normalization
        # use absolute trick since we will revert zeros later on
        IF_mat = np.abs(IF_mat)
        # revert zeros
        IF_mat = IF_mat * idx_mat
        # fix any potential Infs or NaN's
        IF_mat[np.isnan(IF_mat)] = 0
        IF_mat[np.isinf(IF_mat)] = 0
        # rebuild table
        norm_tab = tab.copy()
        norm_tab.iloc[:, 4:] = IF_mat
        return norm_tab
    def _fill_linear_interpolate(self, arr):
        """
        Fill NaN values in a 1-D array with linearly increasing or decreasing values based on its nearest non-NaN values.
        """
        mask = np.isnan(arr)
        idx = np.arange(len(arr))
        arr[mask] = np.interp(idx[mask], idx[~mask], arr[~mask])
        return arr
    def normalize(self):
        # slope_cut for using different lowess
        slope_cut = 0.5
        # step1: divide into groups based on the number digestion
        print("Step1: Separation based on the number of digestion sites..")
        current_md_gb = self.current_md.groupby("#Digsites")
        # step2: use polyfit degree1 to estimate the slope, if the slope < 0.6 and > 1.4, then filtered for lowess normalization
        # from step2, all bins are divide into lowess part and non-lowess part
        print("Step2: PolyFit filtering")
        lowess_lists_bd = []  # store for big difference
        lowess_lists_sd = [] # store for small difference
        gb_lines_dict = {}
        for dig_g, dig_g_df in current_md_gb:
            if dig_g == 0:
                # we don't use digsite = 0 bins
                continue
            else:
                if len(dig_g_df) >= 2 and dig_g_df['count_HiC'].sum() != 0 and dig_g_df['count_MiC'].sum() != 0:
                    # remove outliers
                    hic_outliers = self._is_outlier(dig_g_df['count_HiC'].values,5)
                    microc_outliers = self._is_outlier(dig_g_df['count_MiC'].values, 5)
                    dig_g_df_rm = dig_g_df[~(hic_outliers | microc_outliers)]
                    
                    # Early check: need >= 2 points AND variance in HiC for polyfit
                    if len(dig_g_df_rm) < 2:
                        # Not enough points, add to big difference (will get LOESS normalized)
                        lowess_lists_bd.append(dig_g_df.index.values)
                        continue
                    
                    hic_std = np.std(dig_g_df_rm['count_HiC'].values)
                    mic_std = np.std(dig_g_df_rm['count_MiC'].values)
                    
                    if hic_std < 1e-10:
                        # No variance in HiC - can't fit line, but can infer relationship
                        if dig_g_df_rm['count_MiC'].sum() > dig_g_df_rm['count_HiC'].sum():
                            # MicroC >> HiC (e.g., HiC=[0,0], MiC=[4,6]) → big difference
                            lowess_lists_bd.append(dig_g_df.index.values)
                        else:
                            # Both low or HiC >= MiC → treat as small difference
                            lowess_lists_sd.append(dig_g_df.index.values)
                        continue
                    
                    try:
                        gb_line = np.polyfit(dig_g_df_rm['count_HiC'], dig_g_df_rm['count_MiC'], 1)
                    except (ValueError, np.linalg.LinAlgError, SystemError):
                        # SystemError is the actual exception raised (ValueError is just __cause__)
                        print("PolyFit failed due to numerical issues. Trying with epsilon...")
                        try:
                            gb_line = np.polyfit(dig_g_df_rm['count_HiC'].values + 0.1, dig_g_df_rm['count_MiC'], 1)
                        except (ValueError, np.linalg.LinAlgError, SystemError):
                            # Manual fallback: simple linear regression formula
                            hic_vals = dig_g_df_rm['count_HiC'].values.astype(np.float64)
                            mic_vals = dig_g_df_rm['count_MiC'].values.astype(np.float64)
                            hic_mean, mic_mean = np.mean(hic_vals), np.mean(mic_vals)
                            denom = np.sum((hic_vals - hic_mean) ** 2)
                            if denom > 1e-10:
                                slope = np.sum((hic_vals - hic_mean) * (mic_vals - mic_mean)) / denom
                                gb_line = np.array([slope, mic_mean - slope * hic_mean])
                            else:
                                # Cannot fit, treat as big difference and skip
                                lowess_lists_bd.append(dig_g_df.index.values)
                                continue

                    gb_lines_dict[dig_g] = gb_line
                    if abs(gb_line[0]) < slope_cut:
                        lowess_lists_sd.append(dig_g_df.index.values)
                    else:
                        lowess_lists_bd.append(dig_g_df.index.values)
                else:
                    lowess_lists_bd.append(dig_g_df.index.values)
        # step3: lowess normalize the lowess part
        print("Step3: LOESS Normalization..")
        try:
            md4lowess_bd = self.current_md.iloc[np.concatenate(lowess_lists_bd), :]
            md_normed_bd = self._lowessnorm(md4lowess_bd)
        except ValueError:
            md_normed_bd = pd.DataFrame(columns=['Chromosome','Start','End','#Digsites','count_MiC','count_HiC'])
        try:
            md4lowess_sd = self.current_md.iloc[np.concatenate(lowess_lists_sd), :]
            md_normed_sd = self._lowessnorm(md4lowess_sd)
        except ValueError:
            md_normed_sd = pd.DataFrame(columns=['Chromosome','Start','End','#Digsites','count_MiC','count_HiC'])
        md_normed = pd.concat([md_normed_sd, md_normed_bd])
        md_normed = md_normed.sort_index()
        if len(md_normed) != 0:
            print(f"Total {len(md_normed)} Anchors be found")
        else:
            print("Cannot find Any Anchors. Please consider increasing sequence depth")
            exit(1)
        # step4: get regional coefficient for non-lowess part by averaging the lowess part coefficient
        print("Step4: Linear interplotes bins with 0 Digest Enzyme..")
        current_md_normed = self.current_md.copy()
        current_md_normed[['count_HiC_norm', 'count_MiC_norm']] = md_normed[['count_HiC', 'count_MiC']]
        current_md_normed['HiC_coef'] = (current_md_normed['count_HiC_norm'] + 1) / (current_md_normed['count_HiC'] + 1)
        current_md_normed['MiC_coef'] = (current_md_normed['count_MiC_norm'] + 1) / (current_md_normed['count_MiC'] + 1)
        current_md_normed['HiC_coef'] = self._fill_linear_interpolate(current_md_normed['HiC_coef'].values.copy())
        current_md_normed['MiC_coef'] = self._fill_linear_interpolate(current_md_normed['MiC_coef'].values.copy())
        # step5: get all normed bins' count
        current_md_normed['count_HiC_norm1'] = current_md_normed['HiC_coef'] * (current_md_normed['count_HiC'] + 1) - 1
        current_md_normed['count_MiC_norm1'] = current_md_normed['MiC_coef'] * (current_md_normed['count_MiC'] + 1) - 1
        # make up specific case: coef = 0.5 and count = 1, this lead to 0
        current_md_normed.loc[
            (current_md_normed['count_MiC_norm1'] == 0) & (current_md_normed['count_MiC'] != 0), 'count_MiC_norm1'] = \
            current_md_normed.loc[
                (current_md_normed['count_MiC_norm1'] == 0) & (current_md_normed['count_MiC'] != 0), 'MiC_coef'].values
        current_md_normed.loc[
            (current_md_normed['count_HiC_norm1'] == 0) & (current_md_normed['count_HiC'] != 0), 'count_HiC_norm1'] = \
            current_md_normed.loc[
                (current_md_normed['count_HiC_norm1'] == 0) & (current_md_normed['count_HiC'] != 0), 'HiC_coef'].values
        idx_mat = np.where(current_md_normed[['count_MiC', 'count_HiC']].values != 0, 1, 0)
        # in case still some negative values
        current_md_normed[['count_MiC_norm1', 'count_HiC_norm1']] = np.abs(
            current_md_normed[['count_MiC_norm1', 'count_HiC_norm1']].values * idx_mat)
        current_md_normed = current_md_normed.iloc[:, [0, 1, 2, 3, -1, -2]]
        current_md_normed.columns = self.current_md.columns
        return current_md_normed

