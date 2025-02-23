import numpy as np
import pandas as pd
import cooler
import pyranges as pr
from Nuc3DMap_AnchorNorm import AnchorNorm, MDplot
from scipy.sparse import csr_matrix,eye,coo_matrix,triu
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from numba import njit

# based on cooler version
# build potential anchor pool
def bins_with_digestion(current_bins,digest_chr_df):
    current_bins = current_bins.copy()
    current_bins.columns = ['Chromosome','Start','End']
    current_bins['bin_id'] = current_bins.index.values
    current_digest_pr = pr.PyRanges(digest_chr_df)
    current_bins_pr = pr.PyRanges(current_bins)
    current_inter = current_bins_pr.join(current_digest_pr, report_overlap=True).as_df()
    current_inter['#Digsites'] = 1
    anchors_count = current_inter[['bin_id', '#Digsites']].groupby('bin_id').agg({'#Digsites': np.sum}).reset_index()
    current_bins = pd.merge(current_bins,anchors_count,on='bin_id',how='left')
    current_bins['#Digsites'].fillna(0,inplace=True)
    return current_bins

def assign_digestion(pixels_df, bins_df):
    current_pixels = pd.merge(pixels_df, bins_df[['bin_id', '#Digsites']], left_on='bin1_id',
                              right_on='bin_id')
    current_pixels = pd.merge(current_pixels, bins_df[['bin_id', '#Digsites']], left_on='bin2_id',
                              right_on='bin_id')
    current_pixels['#Digsites'] = current_pixels['#Digsites_x'] + current_pixels['#Digsites_y']
    current_pixels = current_pixels[['bin1_id', 'bin2_id', 'count', '#Digsites']]
    return current_pixels

def calMarginCount(coo_chr):
    coo_chr_melt = pd.melt(coo_chr, id_vars='count', value_vars=['bin1_id', 'bin2_id'])
    coo_chr_melt = coo_chr_melt[['count', 'value']]
    coo_chr_melt.columns = ['count', 'bin_id']
    coo_chr_melt = coo_chr_melt.groupby('bin_id').agg({'count': np.sum}).reset_index()
    coo_diagonal = coo_chr[coo_chr['bin1_id'] == coo_chr['bin2_id']]
    coo_diagonal_dict = dict(zip(coo_diagonal['bin1_id'],coo_diagonal['count']))
    # double count the values on the diagonal
    coo_chr_melt['dup_count'] = coo_chr_melt['bin_id'].map(coo_diagonal_dict)
    coo_chr_melt.fillna(0,inplace=True)
    coo_chr_melt['count'] = (coo_chr_melt['count'] - coo_chr_melt['dup_count'])
    return coo_chr_melt[['bin_id','count']]

@njit
def biasMultiply(rows,cols,data,bias):
    data_norm = np.zeros(len(data))
    for i in range(len(data)):
        row, col, val = rows[i], cols[i], data[i]
        data_norm[i] = val * bias[row] * bias[col]
    return data_norm

class MatBalance(object):
    def __init__(self, intrachr_coo_mat, normed_bins_df, balance_method = 'kr_fancy'):
        # balance method can be kr_fancy/kr_simple/vc
        self.coo_mat = intrachr_coo_mat
        self.normed_bins_df = normed_bins_df
        self.balance_method = balance_method
    def _bnewts_csr(self, A, m, tol=1e-6, x0=None, delta=0.1, fl=1, Delta=3, max_iter=1500):
        """
        BNEWTS A balancing algorithm for symmetric matrices

        X = BNEWTS(A,M) attempts to find a vector X such that
        diag(X)*A*diag(X) has row and column sums matching M. A must
        be symmetric and nonnegative. By default, M is uniform and the
        scaled matrix is doubly stochastic.

        X = BNEWTS(A,M,TOL) specifies the tolerance of the method.
        If TOL = [] then BNEWTS uses the default, 1e-6.

        X = BNEWTS(A,M,TOL,X0) specifies the initial guess. If none
        is given then BNEWTS uses the default, vectors of ones.

        X = BNEWTS(A,M,TOL,X0,DELTA) determines how close we are
        willing to allow our balancing vectors can get to the edge of the
        positive cone. We use a relative measure on the size of elements. The
        default value for DEL is 0.1

        X = BNEWTS(A,M,TOL,X0,DELTA,FL) will output intermediate convergence
        statistics if FL = 1. The default value for FL is 1.

        [X, RES] = BNEWTS(A,M,TOL,X0,DELTA,FL) returns the residual error, too.

        BNEWTS is based on a Newton step to solve
        diag(X)*A*diag(X) - M = 0 as an outer iteration. This can be written as
        x_new = (A + diag((A*x_old)./(xold)))\(A*x_old + m./x_old).
        We solve this step approximately by using CG as an iteration and with
        preconditioner D = diag(xold) applied to both sides of the system.

        The iteration continues until the residual is smaller than TOL.
        The residual is measured by norm(diag(x)*A*x - m,2).
        """
        n = len(m)
        e = np.ones(n)
        res = []
        if x0 is None:
            x0 = e
        g = 0.9
        etamax = 0.1
        eta = etamax
        stop_tol = tol * 0.5
        rt = tol ** 2
        x = x0
        v = x * ((A.dot(x)))
        rk = m - v
        rho_km1 = np.dot(rk, rk)
        rout = rho_km1
        rold = rout
        MVP = 0
        i = 0
        if fl == 1:
            print('it    in. it    res')
        iteration = 0
        while rout > rt and iteration <= max_iter:
            i = i + 1
            k = 0
            y = e
            innertol = max([eta ** 2 * rout, rt])
            while rho_km1 > innertol:
                k = k + 1
                if k == 1:
                    Z = rk / v
                    p = Z
                    rho_km1 = np.dot(rk, Z)
                else:
                    beta = rho_km1 / rho_km2
                    p = Z + beta * p
                # update w
                w = x * (A.dot(x * p)) + v * p
                alpha = rho_km1 / np.dot(p, w)
                ap = alpha * p
                ynew = y + ap
                # new y compare with delta
                if np.min(ynew) <= delta:
                    if delta == 0:
                        break
                    ind = np.where(ap < 0)[0]
                    gamma = np.min((delta - y[ind]) / ap[ind])
                    y = y + gamma * ap
                    break
                # new y larger than delta
                if np.max(ynew) >= Delta:
                    ind = np.where(ynew > Delta)[0]
                    gamma = np.min((Delta - y[ind]) / ap[ind])
                    y = y + gamma * ap
                    break
                # update inner iters
                y = ynew
                rk = rk - alpha * w
                rho_km2 = rho_km1
                Z = rk / v
                rho_km1 = rk.T @ Z
            # update outer iters
            x = x * y
            v = x * ((A.dot(x)))
            rk = m - v
            rho_km1 = rk.T @ rk
            rout = rho_km1
            MVP = MVP + k + 1
            rat = rout / rold
            rold = rout
            r_norm = np.sqrt(rout)
            eta_o = eta
            eta = g * rat
            if g * eta_o ** 2 > 0.1:
                eta = max([eta, g * eta_o ** 2])
            eta = max([min([eta, etamax]), stop_tol / r_norm])
            if fl == 1:
                print(f'{i:3d} {k:6d}   {r_norm:.3e} {np.min(y):.3e} {np.min(x):.3e}')
            if iteration == max_iter:
                print(f"Reach maximum iterations number: {max_iter}, stopped")
            res.append(r_norm)
            iteration += 1
        return x, res
    def adjKR_fancy(self,matsize):
        coo_mat = self.coo_mat.copy()
        marginSum_df = self.normed_bins_df.copy()
        csr_mat = coo_mat.tocsr()
        # remove zeros columns and rows in csr_mat
        nz = csr_mat.getnnz(1) > 0
        B = csr_mat[nz]
        B = B[:, nz]
        # quick check marginal value length = matrix size
        if B.shape[0] != len(marginSum_df):
            print(f"Length Unmatched Error:{B.shape[0]} != {len(marginSum_df)}")
            exit(1)
        y = marginSum_df['count'].values
        # find connected_component (permute the matrix)
        n_components, cc = connected_components(B, directed=False, return_labels=True)
        nblk = len(np.unique(cc))
        p = np.zeros(len(marginSum_df))
        allres = []
        for i in range(nblk):
            indices = np.where(cc == i)[0]
            l = len(indices)
            # eps helps the convergence of the balancing
            sub_matrix = B[np.ix_(indices, indices)] + eye(l, format='csc') * np.finfo(np.float32).eps
            sub_y = y[indices]
            sub_p, res = self._bnewts_csr(sub_matrix, sub_y, tol=1e-6, x0=np.ones(l), delta=0.1, fl=0)
            allres.append(res)
            p[indices] = sub_p
        x = np.ones(matsize)
        x[nz] = p
        return allres, x
    def normCoo(self, biasArray):
        coo_mat = self.coo_mat.copy()
        data_normed = biasMultiply(coo_mat.row,coo_mat.col,coo_mat.data,biasArray)
        coo_mat_normed = coo_matrix((data_normed,(coo_mat.row,coo_mat.col)),shape=(coo_mat.shape[0],coo_mat.shape[1]))
        return coo_mat_normed

def chrom_norm(microc_clr,hic_clr,digest_chrs,ref_dict,prefix,outdir,plotMD=False):
    # todo:might use yield to save space
    hic_coos = []
    microc_coos = []
    bins_list = []
    res_dict = {'HiC':{},'MicroC':{}}
    for chr in ref_dict:
        print(f"Processing {chr}")
        bins_chr = microc_clr.bins().fetch(chr)
        bin_chr = bins_chr[['chrom', 'start', 'end']]
        # find bins with digesetion sites
        current_bins = bins_with_digestion(bin_chr, digest_chrs[chr])
        # matrix return full matrix but pixels return tiru
        microc_coo_chr = microc_clr.matrix(balance=False,sparse=True).fetch(chr)
        hic_coo_chr = hic_clr.matrix(balance=False,sparse=True).fetch(chr)
        relevel_id = current_bins['bin_id'].min()
        # calculate marginalizedContactCount
        microc_coo_chr_margin = microc_coo_chr.sum(axis=1).A1
        hic_coo_chr_margin = hic_coo_chr.sum(axis=1).A1
        current_bins['count_MiC'] = microc_coo_chr_margin
        current_bins['count_HiC'] = hic_coo_chr_margin
        current_md = current_bins[['Chromosome', 'Start', 'End', '#Digsites', 'count_MiC', 'count_HiC']]
        # filter counts equal zero
        current_md = current_md[~(current_md[['count_MiC', 'count_HiC']].sum(axis=1) == 0)]
        current_md.reset_index(drop=True, inplace=True)
        anchornorm = AnchorNorm(current_md) # if #Digsites is 0, normed by constant coef sum(count_HiC)/sum(count_MiC)
        current_md_normed = anchornorm.normalize()
        # visualize the effect of the normalization
        if plotMD:
            print("Visualizing the margin correction")
            MDplot(current_md[~(current_md[['#Digsites','count_HiC','count_MiC']]== 0).any(axis=1)], dcol='#Digsites',
                   outdir=f'{outdir}/{prefix}_{chr}_margin_raw.png', neighbours=20, plot_loess=True,
                   plot_scatter=True, plot_outliers=False)
            MDplot(current_md_normed[~(current_md_normed[['#Digsites','count_HiC','count_MiC']]== 0).any(axis=1)], dcol='#Digsites',
                   outdir=f'{outdir}/{prefix}_{chr}_margin_normed.png', neighbours=20, plot_loess=True,
                   plot_scatter=True, plot_outliers=False)
        # add ID information
        current_md_normed = pd.merge(current_md_normed, current_bins[['Chromosome', 'Start', 'End', 'bin_id']],
                                     on=['Chromosome', 'Start', 'End'])
        # matrix balancing according to the marginal summation
        matsize = len(current_bins)
        print("Matrix balancing for HiC")
        hic_NormedSum = current_md_normed.loc[current_md_normed['count_HiC']!=0,['bin_id','count_HiC']]
        hic_NormedSum.columns = ['bin_id','count']
        # relevel the coor
        hic_NormedSum['bin_id'] -= relevel_id
        hicMatBalance = MatBalance(hic_coo_chr,hic_NormedSum)
        # KR fancy normalization
        hicRes_krf, hicBias = hicMatBalance.adjKR_fancy(matsize)
        current_bins['HiC_Bias'] = hicBias
        hic_coo_chr_krf_normed = hicMatBalance.normCoo(current_bins['HiC_Bias'].values)
        res_dict['HiC'][chr] = hicRes_krf
        print("Matrix balancing for MicroC")
        microc_NormedSum = current_md_normed.loc[current_md_normed['count_MiC'] != 0, ['bin_id', 'count_MiC']]
        microc_NormedSum.columns = ['bin_id', 'count']
        # relevel the coo
        microc_NormedSum['bin_id'] -= relevel_id
        microcMatBalance = MatBalance(microc_coo_chr, microc_NormedSum)
        # KR fancy normalization
        microcRes_krf, microcBias = microcMatBalance.adjKR_fancy(matsize)
        current_bins['MicroC_Bias'] = microcBias
        microc_coo_chr_krf_normed = microcMatBalance.normCoo(current_bins['MicroC_Bias'].values)
        res_dict['MicroC'][chr] = microcRes_krf
        # stored in a list for creating cooler
        hic_coos.append(hic_coo_chr_krf_normed)
        microc_coos.append(microc_coo_chr_krf_normed)
        current_bins = current_bins[['Chromosome', 'Start', 'End', 'bin_id', '#Digsites']]
        current_bins.columns = ['chrom', 'start', 'end', 'bin_id', '#Digsites']
        bins_out = pd.merge(current_bins, bins_chr[['chrom','start','end','nuc','Itype','spacing','positioning']],
                            on=['chrom', 'start', 'end'], how='left')
        bins_list.append(bins_out)
    return res_dict, hic_coos, microc_coos, bins_list

@njit
def removeDigestZeros(rows,cols,data,numDigest):
    data_filt = []
    rows_filt = []
    cols_filt = []
    for i in range(len(data)):
        row, col, val = rows[i], cols[i], data[i]
        if numDigest[row] + numDigest[col] != 0:
            rows_filt.append(row)
            cols_filt.append(col)
            data_filt.append(val)
        else:
            continue
    return rows_filt,cols_filt,data_filt

def iMHiC(microc_coos, hic_coos, bins_list,ref_dict):
    imhic_coos = []
    i = 0
    for chrom in ref_dict:
        print(f'Processing {chrom}')
        bins_chr = bins_list[i]
        relevel_id = bins_chr['bin_id'].min()
        hic_coo_chr = hic_coos[i]
        # remove contact count on 0 Digsites in HiC data
        hic_rows_filt,hic_cols_filt,hic_data_filt = removeDigestZeros(hic_coo_chr.row,hic_coo_chr.col,
                                                                      hic_coo_chr.data,bins_chr['#Digsites'].values)
        hic_coo_chr_filt = coo_matrix((hic_data_filt,(hic_rows_filt,hic_cols_filt)),
                                      shape=(hic_coo_chr.shape[0],hic_coo_chr.shape[1]))

        # integrate MicroC with HiC with rules: 1. use MicroC if common, 2. makeup MicroC with HiC_uniq
        microc_coo_chr = microc_coos[i]

        hic_filt_pairs = set(zip(hic_coo_chr_filt.row,hic_coo_chr_filt.col))
        microc_pairs = set(zip(microc_coo_chr.row, microc_coo_chr.col))
        # find uniq pairs
        hic_filt_uniq_pairs = np.array(list(hic_filt_pairs - microc_pairs))

        # Unpack unique rows into separate arrays
        hic_filt_uniq_row = hic_filt_uniq_pairs[:, 0]
        hic_filt_uniq_col = hic_filt_uniq_pairs[:, 1]

        hic_csr_chr_filt = hic_coo_chr_filt.tocsr()
        hic_filt_uniq_data = hic_csr_chr_filt[hic_filt_uniq_row,hic_filt_uniq_col].A1

        hic_coo_chr_filt_uniq = coo_matrix((hic_filt_uniq_data,(hic_filt_uniq_row,hic_filt_uniq_col)),
                                           shape=(microc_coo_chr.shape[0],microc_coo_chr.shape[1]))
        # integrate two matrices
        imhic_coo_chr = hic_coo_chr_filt_uniq + microc_coo_chr
        # add weight to bins resulted from conventional KR norm
        # extract non-zero rows
        imhic_coo_chr_margin = imhic_coo_chr.sum(axis=1).A1
        imhic_coo_chr_melt = pd.DataFrame({
            'bin_id': relevel_id + np.nonzero(imhic_coo_chr_margin)[0],
            'count_MiC': imhic_coo_chr_margin[imhic_coo_chr_margin != 0]
        })
        imhic_coo_chr_melt_nnz = imhic_coo_chr_melt.copy()
        imhic_coo_chr_melt_nnz['count'] = 1
        imhic_coo_chr_melt_nnz['bin_id'] -= relevel_id
        matsize = len(bins_chr)
        imhicMatBalance = MatBalance(imhic_coo_chr, imhic_coo_chr_melt_nnz)
        # KR fancy normalization
        imhicRes_krf, imhicBias = imhicMatBalance.adjKR_fancy(matsize)
        bins_chr['weight'] = imhicBias
        bins_list[i] = bins_chr
        imhic_coos.append(imhic_coo_chr)
        i += 1
    return bins_list, imhic_coos

def visualConverge(ref_dict, res_dict, prefix, outdir):
    # visualize the convergence
    fig, axs = plt.subplots(6, 4, figsize=(20, 20))
    chr_idx = 0
    for chrom in ref_dict:
        idx = chr_idx // 4
        idy = chr_idx % 4
        ax = axs[idx][idy]
        hic_krf_res = res_dict['HiC'][chrom]
        # pick the res for the largest connected matrix
        hic_krf_res = hic_krf_res[hic_krf_res.index(max(hic_krf_res, key=len))]
        microc_krf_res = res_dict['MicroC'][chrom]
        # pick the res for the largest connected matrix
        microc_krf_res = microc_krf_res[microc_krf_res.index(max(microc_krf_res, key=len))]
        ax.plot(np.arange(len(hic_krf_res)), hic_krf_res, label='HiC', color='orange')
        ax.plot(np.arange(len(microc_krf_res)), microc_krf_res, label='MicroC', color='purple')
        ax.set_yscale('log')
        ax.set_title(f'{chrom}')
        chr_idx += 1
    legend = ax.legend(loc='upper right', fontsize='x-large')
    plt.xlabel('iterations')
    plt.ylabel('residuals')
    plt.tight_layout()
    plt.savefig(f'{outdir}/{prefix}_MHiC_convergence.png', dpi=300)
    plt.close()

def getpixels(coo_mat_triu,relevel_binid,relevel_dfid):
    pixels_df = pd.DataFrame({
        'bin1_id':coo_mat_triu.row + relevel_binid,
        'bin2_id':coo_mat_triu.col + relevel_binid,
        'count':coo_mat_triu.data
    })
    pixels_df.index = relevel_dfid + pixels_df.index.values
    return pixels_df

def chrom_iterator(coo_list,bins_list,ref_dict):
    relevel_df_idx = 0
    for idx,chr in enumerate(list(ref_dict.keys())):
        print(chr)
        coo_mat = coo_list[idx]
        coo_mat_triu = triu(coo_mat)
        relevel_binid = bins_list[idx]['bin_id'].min()
        yield getpixels(coo_mat_triu,relevel_binid,relevel_df_idx)
        relevel_df_idx += len(coo_mat_triu.data)

def nucmerge_save_cooler(cooler_path,bins_list,coo_list,ref_dict):
    chrom_iter = chrom_iterator(coo_list,bins_list,ref_dict)
    bins_df = pd.concat(bins_list)
    cooler.create_cooler(cool_uri=cooler_path,
                         bins=bins_df,
                         pixels=chrom_iter,
                         ordered=True,
                         dtypes={'count': np.float32})
    return None







