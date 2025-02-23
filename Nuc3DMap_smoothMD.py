import pandas as pd
import numpy as np
import matplotlib.cm as cm
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import statsmodels.api as sm
from matplotlib.colors import LogNorm

def data_coord2view_coord(p, resolution, pmin, pmax):
    dp = pmax - pmin
    dv = (p - pmin) / dp * resolution
    return dv

def nMAD(x,n=6):
    xm = np.median(x)
    md = np.median(np.abs(x-xm))
    mads_up = xm + n * md
    mads_dw = xm - n * md
    return mads_up,mads_dw

def kNN2DDens(xv, yv, resolution, neighbours, dim=2):
    """

    :param xv:
    :param yv:
    :param resolution:
    :param neighbours:
    :param dim:
    :return:
    """
    # Create the tree
    tree = cKDTree(np.array([xv, yv]).T)
    # Find the closest nnmax-1 neighbors (first entry is the point itself)
    grid = np.mgrid[0:resolution, 0:resolution].T.reshape(resolution**2, dim)
    dists = tree.query(grid, neighbours)
    # Inverse of the sum of distances to each grid point.
    sum_dists = dists[0].sum(1)
    zero_indices = sum_dists == 0
    sum_dists[zero_indices] = 1
    inv_sum_dists = 1. / sum_dists
    inv_sum_dists[zero_indices] = 0
    # Reshape
    im = inv_sum_dists.reshape(resolution, resolution)
    return im

# plot MD plot
def MD_smoothing(M, D, p_val=None, title='MD Plot', resolution=250, neighbours=20, ylab = 'M', xlab = 'Distance',
                 plot_loess=False, plot_scatter=False, plot_outliers=False, outlier_n=6,
                 D_range=1, ax=None):
    # subset plot by D
    if D_range < 1:
        max_D = np.max(D)
        cut_point = np.ceil(D_range * max_D)
        # subset M, D, p_val vectors based on cut_point
        keep = D <= cut_point
        D = D[keep]
        M = M[keep]
        p_val = p_val[keep]
    # smooth scatter version
    extent = [np.min(D), np.max(D), np.min(M), np.max(M)]
    xv = data_coord2view_coord(D, resolution, extent[0], extent[1])
    yv = data_coord2view_coord(M, resolution, extent[2], extent[3])
    im = kNN2DDens(xv, yv, resolution, neighbours)
    # in case im is all zeros, change neighbours
    while len(im[im>0])==0:
        neighbours -= 2
        if neighbours > 0:
            im = kNN2DDens(xv, yv, resolution, neighbours)
        else:
            print("Cannot plot MD-Plot due to kernel issue, skipped")
            return None
    ax.imshow(im, origin='lower', extent=extent, cmap=cm.Blues, aspect='auto',
              norm=LogNorm(vmin=np.min(im), vmax=np.max(im)))
    if plot_outliers:
        # deprecated
        # detect potential outliers for plot
        # indices = pd.DataFrame(np.argwhere(im < nMAD(im, outlier_n)))
        # xinterval = np.linspace(extent[0],extent[1],resolution)
        # yinterval = np.linspace(extent[2],extent[3],resolution)
        # interval_df = pd.DataFrame({'x':np.linspace(extent[0],extent[1],resolution),
        #                              'y':np.linspace(extent[2],extent[3],resolution)})
        # interval_df['xID'] = interval_df.index.values
        # interval_df['xinterval'] = pd.cut(interval_df['x'],xinterval,include_lowest=True)
        # interval_df['yinterval'] = pd.cut(interval_df['y'],yinterval,include_lowest=True)
        # xintervalID = dict(zip(interval_df['xinterval'],interval_df['xID']))
        # yintervalID = dict(zip(interval_df['yinterval'],interval_df['xID']))
        # MD = pd.DataFrame({'M':M,'D':D})
        # MD['xinterval'] = pd.cut(MD['D'],xinterval,include_lowest=True)
        # MD['yinterval'] = pd.cut(MD['M'],yinterval,include_lowest=True)
        # MD['xID'] = MD['xinterval'].map(xintervalID)
        # MD['yID'] = MD['yinterval'].map(yintervalID)
        # MD_filt = pd.merge(MD,indices,left_on=['xID','yID'],right_on=[0,1])
        MD = pd.DataFrame({'M': M, 'D': D})
        MD_filt_list = []
        for d,md_d in MD.groupby('D'):
            if len(md_d) > 10:
                mads_up,mads_dw = nMAD(md_d['M'],outlier_n)
                md_d_filt = md_d[(md_d['M']<mads_dw)|(md_d['M']>mads_up)]
                MD_filt_list.append(md_d_filt)
        MD_filt = pd.concat(MD_filt_list)
        ax.plot(MD_filt['D'], MD_filt['M'], 'k.', markersize=1)
    else:
        if plot_scatter:
            ax.plot(D, M, 'k.', markersize=3)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.plot([np.min(D),np.max(D)], [0,0], 'k-', linewidth=2)
    if p_val is not None:
        if not np.isnan(p_val[0]):
            p0_001 = np.where(p_val < 0.001)[0]
            p0_05 = np.where((p_val >= 0.001) & (p_val < 0.05))[0]
            p0_001_scatter = ax.scatter(D[p0_001], M[p0_001], color='red', marker='o')
            p0_05_scatter = ax.scatter(D[p0_05], M[p0_05], color='yellow', marker='o')
            ax.legend([p0_001_scatter,p0_05_scatter],['P < 0.001', 'P < 0.05'], loc='lower right')
    # add loess fit to plot
    if plot_loess:
        loess = sm.nonparametric.lowess(endog=M, exog=D)
        try:
            f = interp1d(loess[:,0], loess[:,1], kind='cubic')
            xnew = np.linspace(min(loess[:,0]), max(loess[:,0]), num=1000, endpoint=True)
            ax.plot(xnew, f(xnew), 'r-', linewidth=2)
        except ValueError:
            loess_df = pd.DataFrame(loess)
            loess_df.drop_duplicates(inplace=True)
            ax.plot(loess_df[0], loess_df[1], 'r-', linewidth=2)