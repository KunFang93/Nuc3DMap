import sys, os
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import seaborn as sns
import matplotlib.pyplot as plt
import pyranges as pr
import datetime
import cooler
from Nuc3DMap_utilities import GenomicAnnot

############# NucintervalBin Start ####################
def get_time():
    # get the current time
    now = datetime.datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")
    return dt_string
class PrepNuc(object):
    def __init__(self, nuc_f, ref_dict, gene_f, prefix='Interval', outdir=os.getcwd(),
                 extsize = 35, min_interval = 10, interval_binsize=300, max_spacing=10000, plot_mark=False):
        self.nuc_f = nuc_f
        self.ref_dict = ref_dict
        self.gene_annot = pd.read_table(gene_f, header=None, names=['Chromosome', 'Start', 'End', 'Strand', 'GeneName', 'Count'])
        self.extsize = extsize
        self.min_interval = min_interval
        self.interval_binsize = interval_binsize
        self.plot_mark = plot_mark
        self.prefix = prefix
        self.outdir = outdir
        self.max_spacing = max_spacing
    def _fillinterval(self,nuc_chr_df):
        chrom = nuc_chr_df[0].values[0]
        interval_chr_df = pd.DataFrame(
            {'Chromosome': [chrom] * (len(nuc_chr_df) + 1),
             'Start': [0] + nuc_chr_df['bin_end'].values.tolist(),
             'End': nuc_chr_df['bin_start'].values.tolist() + [self.ref_dict[chrom]],
             'pvalley_l': [20] + nuc_chr_df[9].values.tolist(),
             'pvalley_r': nuc_chr_df[9].values.tolist() + [20]
             })
        return interval_chr_df
    def _nuc_pos_degree(self, nuc_info_df, k=1.5):
        '''
        calculate nucleosome position score and categorize it
        :param nuc_info_df:
        :param k:
        :param degree_of_class: how many classes to mark the nucleosome positioning, e.g. 0 low position score, 1 median position score..
        :return:
        '''
        def outlier_threshold(target_list, k):
            '''
            Use q3 + k * iqr to identify outlier
            :param target_list:
            :param k: coefficient
            :return: up cutoff for the target_list
            '''
            data = target_list
            quartile1 = np.quantile(data, 0.25)
            quartile3 = np.quantile(data, 0.75)
            interval_qr = quartile3 - quartile1
            upb = quartile3 + k * interval_qr
            return upb
        width_up_limit = outlier_threshold(nuc_info_df[4], k)
        height_up_limit = outlier_threshold(nuc_info_df[5], k)
        area_up_limit = outlier_threshold(nuc_info_df[6], k)
        pp_up_limit = outlier_threshold(nuc_info_df[8], k)
        pv_up_limit = outlier_threshold(nuc_info_df[9], k)
        nucs_pvalvalley = nuc_info_df[9][:]
        nucs_pvalvalley[nucs_pvalvalley > pv_up_limit] = np.mean(nucs_pvalvalley)
        pv_up_limit = outlier_threshold(nucs_pvalvalley, k)
        print('Calculating the positioning score..')
        print('Positioning score equation is (height+log2(PP*PV+1)+sqrt(area))/(width*3).')
        # transfer to nd-array for calculation and normalize it
        width_norm = np.array(nuc_info_df[4]) / width_up_limit
        height_norm = np.array(nuc_info_df[5]) / height_up_limit
        area_norm = np.array(nuc_info_df[6]) / area_up_limit
        pvalpeak_norm = np.array(nuc_info_df[8]) / pp_up_limit
        pvalvalley_norm = np.array(nucs_pvalvalley) / pv_up_limit
        # calculate the positioning scores
        positioning_score = (height_norm + np.log2(pvalpeak_norm * pvalvalley_norm + 1) + area_norm) / (width_norm * 3)
        # final scaling
        score_up_limit = outlier_threshold(positioning_score,k)
        positioning_score_final = (positioning_score/score_up_limit) * 20
        return positioning_score_final
    def _nuc_spa(self,nuc_df):
        # first find gaps
        spacing_array = ((nuc_df[1] + nuc_df[2]) / 2).diff()
        # make the first nucleosome of each chromosome diff value from negative value to the spacing of the second
        spacing_array[spacing_array[spacing_array <= 0].index] = spacing_array[
            spacing_array[spacing_array <= 0].index.values + 1]
        spacing_array[spacing_array >= self.max_spacing] = self.max_spacing
        # add spacing for the first nucleosome
        spacing_array[0] = spacing_array[1]
        return spacing_array
    def _load_nuc(self):
        print("Loading files")
        print("Loading nucleosome information")
        nuc_df = pd.read_table(self.nuc_f, header=None)
        print("Calculating nucleosome positioning and spacing")
        # assign nucleosome spacing to each nucleosome
        nuc_spacing = self._nuc_spa(nuc_df)
        nuc_df['spacing'] = nuc_spacing
        # assign nucleosome spacing to each nucleosome
        nuc_positioning = self._nuc_pos_degree(nuc_df)
        nuc_df['positioning'] = nuc_positioning
        # remove some shoulders by columns 7 (area)
        ax = sns.kdeplot(data=nuc_df[nuc_df[7] == 'Shoulder'], x=6, log_scale=2)
        x = ax.lines[0].get_xdata()
        y = ax.lines[0].get_ydata()
        # 3/4 of max peak
        half_max_y = np.max(y) * 3 / 4
        left_idx = np.where(y >= half_max_y)[0][0]
        left_idx = min(left_idx, len(y) - 2)
        x_cut = x[left_idx]
        print(f"Removing shoulder with area smaller than {x_cut}")
        nuc_df = nuc_df.loc[~((nuc_df[7] == "Shoulder") & (nuc_df[6] < x_cut)), [0, 1, 2, 3, 9, 'spacing', 'positioning']]
        # extent if nucleosome core size less than 150
        print("Processing nucleosomes")
        nuc_df['bin_start'] = np.where((nuc_df[2] - nuc_df[1]) < 150, nuc_df[1] - self.extsize, nuc_df[1])
        nuc_df['bin_end'] = np.where((nuc_df[2] - nuc_df[1]) < 150, nuc_df[2] + self.extsize, nuc_df[2])
        # avoid nucleosome overlap
        nuc_subdf_list = []
        interval_subdf_list = []
        nuc_df_gb = nuc_df.groupby(0)
        for chr in self.ref_dict:
            # print("Loading nucleosomes in {}".format(chr))
            nuc_gb = nuc_df_gb.get_group(chr)
            nuc_gb['tmp_end'] = [0] + list(nuc_gb['bin_end'])[:-1]
            nuc_gb_adj_bool = (nuc_gb['bin_start'] - nuc_gb['tmp_end']) < self.min_interval
            nuc_gb_adj_coor = nuc_gb[nuc_gb_adj_bool][['bin_start', 'tmp_end']].mean(axis=1)
            nuc_gb.loc[nuc_gb_adj_bool, 'bin_start'] = (nuc_gb_adj_coor + self.min_interval / 2).astype(int)
            nuc_gb.loc[nuc_gb_adj_bool.shift(-1).fillna(False), 'bin_end'] = (nuc_gb_adj_coor - self.min_interval / 2).astype(int).to_list()
            nuc_gb.loc[:, 'bin_end'] = nuc_gb['bin_end'].astype(int)
            nuc_gb.loc[:, 'bin_len'] = nuc_gb['bin_end'] - nuc_gb['bin_start']
            nuc_subdf_list.append(nuc_gb)
            interval_subdf_list.append(self._fillinterval(nuc_gb))
        nuc_ext_df = pd.concat(nuc_subdf_list).reset_index(drop=True)
        nuc_ext_df = nuc_ext_df[[0, 'bin_start', 'bin_end', 3, 9, 'spacing', 'positioning']]
        nuc_ext_df.columns = [0, 1, 2, 3, 9, 'spacing', 'positioning']
        interval_final_df = pd.concat(interval_subdf_list).reset_index(drop=True)
        # plus 1 to accommodate cooler
        interval_final_df['idx'] = ['interval_{}'.format(i) for i in range(1, len(interval_final_df) + 1)]
        return nuc_ext_df, interval_final_df
    def _finalizeInterval(self, interval_df, plot_distribution=False, low_mad_coef=2, high_mad_coef=5,
                          minintervalLen=70, desertcut = 5000, ndgcut = 500):
        """
        Apply filtering and annotating steps for intervals
        :param interval_df: dataframe contains all intervals
        :param pvalley_mad_coef: mad coefficient to calculate pvalley.cut
        :param intervalLen_mad_coef: mad coefficient to calculate intervalLen.cut
        :param minintervalLen: hard intervalLen.cut
        :param plot_distribution: plot pvalley and intervalLen distribution
        :return: dataframe contains final intervals
        """
        # find length cut
        interval_df['len'] = interval_df['End'] - interval_df['Start']
        len_MAD = np.median(np.absolute(interval_df['len'] - np.median(interval_df['len'])))
        # find pvalley.MAD.cut
        all_pvalley = interval_df['pvalley_l'].values[1:]
        # pre-filt too interval with too small pvalley
        all_pvalley_filt = all_pvalley[all_pvalley > np.power(0.5, 5)]
        pvalley_cut = low_mad_coef * np.median(np.absolute(all_pvalley_filt - np.median(all_pvalley_filt)))
        # filter1: filter with hard cut minIntervalLen
        interval_filt = interval_df[interval_df['len']>minintervalLen]
        # filter2: filter with interval with both length and pvalley (length < len_cut and at least one of end have small pvalley)
        interval_filt = interval_filt[
            ~(((interval_filt['pvalley_l'] <= pvalley_cut) | (interval_filt['pvalley_r'] <= pvalley_cut)) &
            (interval_filt['len'] <= max(high_mad_coef * len_MAD, minintervalLen)))
        ]
        # annotate genomic location
        print("Identifying different Interval types")
        intervalannot = GenomicAnnot(interval_filt,self.gene_annot,bidirect=True)
        interval_filt_annot = intervalannot.annot()
        interval_filt = pd.merge(interval_filt,interval_filt_annot,on=['Chromosome','Start','End'], how='left')
        # further filter short interval located in genebody
        interval_filt = interval_filt[~((interval_filt['genomeLoc_annot'].isin(['Genebody']))&(interval_filt['len']<max(3*len_MAD,minintervalLen+20)))]
        # annotation different types: NFRs, NDGs; DRs; NLIR
        interval_filt.loc[(interval_filt['pvalley_l']<pvalley_cut)|(interval_filt['pvalley_r']<pvalley_cut),'Type'] = 'NLIR'
        interval_filt.loc[(interval_filt['genomeLoc_annot']=='Desert')|(interval_filt['len']>desertcut),'Type'] = 'DR'
        interval_filt.loc[(interval_filt['genomeLoc_annot'].isin(['Distal','Proximal','Genebody'])) &
                          (interval_filt['len'] > ndgcut) &
                          (interval_filt['Type'].isna()),'Type'] = 'NDG'
        interval_filt.loc[(interval_filt['Type'].isna()),'Type'] = 'NFR'
        type2id = {'NFR':1,'NDG':2,'DR':3,'NLIR':4}
        interval_filt['Itype'] = interval_filt['Type'].map(type2id)
        # interval_filt[['Chromosome','Start','End','idx','genomeLoc_annot','Type']].to_csv(f'{self.outdir}/{self.prefix}_intervals.{get_time()}.txt',index=False,sep='\t')
        if plot_distribution:
            fig, axs = plt.subplots(1, 2)
            sns.kdeplot(data=interval_df, x='pvalley_r', log_scale=2, ax=axs[0])
            axs[0].axvline(x=pvalley_cut, c='r', linestyle='--')
            axs[0].set_xlabel('Log2(pvalley)')
            sns.kdeplot(data=interval_df, x='len', log_scale=2, ax=axs[1])
            axs[1].axvline(x=high_mad_coef*len_MAD, c='r', linestyle='--')
            axs[1].set_xlabel('Log2(Interval Len)')
            plt.tight_layout()
            plt.savefig(f'{self.outdir}/pvalley_intervalLen_dist.{get_time()}.png', dpi=300)
            plt.close()
        return interval_filt[['Chromosome','Start','End','idx','Itype']]
    def _DivInterval(self,x):
        # Divide the large bin by the following rules:
        # 1. if the length less 300bp, keep it.
        # 2. if the length is larger than 300bp but less than 600, divided by 2.
        # 3. if the length is larger than 600bp less than 300 * 100, divide by 300 but the last two bin is calculated by mean length.
        # 4. if the length is larger than 300*100, then equally divide it into 100 bins. point4 is to reduce the matrix size and save computation resource
        intervallen = x['End'] - x['Start']
        if intervallen < self.interval_binsize:
            start_list = np.array([x['Start']])
            end_list = np.array([x['End']])
        elif intervallen < 2 * self.interval_binsize:
            start_list = np.array([x['Start'], x['Start'] + int(intervallen/2)])
            end_list = np.array([x['Start']+int(intervallen/2), x['End']])
        elif intervallen < 100 * self.interval_binsize:
            binnum = np.ceil(intervallen/self.interval_binsize)
            tmp_start = x['Start'] + np.arange(binnum) * self.interval_binsize
            start_list = np.append(tmp_start[:-1],np.array([int(np.mean([tmp_start[-2],x['End']]))]))
            end_list = np.append(start_list[1:], x['End'])
        else:
            binsize = int(intervallen/100)
            start_list = x['Start'] + np.arange(100) * binsize
            end_list = np.append(start_list[1:],x['End'])
        subidx = np.arange(len(start_list)) / np.power(10, len(str(len(start_list))))
        return [start_list, end_list, subidx]
    def binproc(self):
        nuc_ext_df, interval_df = self._load_nuc()
        # load and process gene annotation file
        print("Loading gene annotation file")
        if self.plot_mark:
            interval_final_df = self._finalizeInterval(interval_df, plot_distribution=True)
        else:
            interval_final_df = self._finalizeInterval(interval_df)
        print("Binarizing intervals")
        interval_final_df[['S-list', 'E-list', 'subidx']] = interval_final_df[
            ['Start', 'End']].apply(lambda x: self._DivInterval(x), axis=1, result_type='expand')
        interval_final_expand_df = interval_final_df.explode(['S-list', 'E-list', 'subidx']).reset_index(drop=True)
        interval_final_expand_df[['Start', 'End']] = interval_final_expand_df[['S-list', 'E-list']].astype(
            int)
        interval_final_expand_df.reset_index(inplace=True, drop=True)
        interval_final_expand_df['Spacing'] = 0
        interval_final_expand_df['Positioning'] = 0
        # process nuc_ext_df
        nuc_final_df = nuc_ext_df[[0, 1, 2, 'spacing','positioning']]
        nuc_final_df.columns = ['Chromosome', 'Start', 'End', 'Spacing', 'Positioning']
        nuc_final_df['idx'] = ['nuc_{}'.format(i) for i in range(1, len(nuc_final_df) + 1)]
        nuc_final_df['Itype'] = 0
        # combine nuc_ext_df and interval_final_expand_df
        final_df = pd.concat([interval_final_expand_df, nuc_final_df])
        final_srt_df = pd.concat([final_df[final_df["Chromosome"]==chrom].sort_values('Start') for chrom in nuc_final_df['Chromosome'].unique()])
        final_srt_df.drop(['S-list', 'E-list', 'subidx'], inplace=True, axis=1)
        final_srt_df.reset_index(inplace=True, drop=True)
        final_srt_df['bin_id'] = final_srt_df.index.values
        final_srt_df['nuc'] = 1
        final_srt_df.loc[final_srt_df['idx'].str.contains('interval'), 'nuc'] = 0
        final_srt_df.drop('idx',axis=1,inplace=True)
        return final_srt_df

############# NucintervalBin Finish ####################
def load_polished_pairs(polished_pairs):
    pairs_df = pd.read_table(polished_pairs)
    pairs_df.columns = ['chrom1', 'pos1.final', 'chrom2', 'pos2.final', 'pos51', 'pos52', 'pos31', 'pos32',
                        'walk_strand_type', 'pattern']
    pairs_df[['strand1', 'strand2']] = pairs_df['walk_strand_type'].str.split(
        '_', expand=True)[1].str.split('/', expand=True)
    return pairs_df

def load_raw_pairs(pairs_path, chunksize):
    _pandas_version = pd.__version__.split('.')
    if int(_pandas_version[0]) > 0:
        from pandas.io.common import get_handle

    # Copied from pairtools._headerops
    def get_header(instream, comment_char='#'):
        '''Returns a header from the stream and an the reaminder of the stream
        with the actual data.
        Parameters
        ----------
        instream : a file object
            An input stream.
        comment_char : str
            The character prepended to header lines (use '@' when parsing sams,
            '#' when parsing pairsams).
        Returns
        -------
        header : list
            The header lines, stripped of terminal spaces and newline characters.
        remainder_stream : stream/file-like object
            Stream with the remaining lines.
        '''
        header = []
        if not comment_char:
            raise ValueError('Please, provide a comment char!')
        comment_byte = comment_char.encode()
        # get peekable buffer for the instream
        read_f, peek_f = None, None
        if hasattr(instream, 'buffer'):
            peek_f = instream.buffer.peek
            readline_f = instream.buffer.readline
        elif hasattr(instream, 'peek'):
            peek_f = instream.peek
            readline_f = instream.readline
        else:
            raise ValueError('Cannot find the peek() function of the provided stream!')
        current_peek = peek_f(1)
        while current_peek.startswith(comment_byte):
            # consuming a line from buffer guarantees
            # that the remainder of the buffer starts
            # with the beginning of the line.
            line = readline_f()
            if isinstance(line, bytes):
                line = line.decode()
            # append line to header, since it does start with header
            header.append(line.strip())
            # peek into the remainder of the instream
            current_peek = peek_f(1)
        # apparently, next line does not start with the comment
        # return header and the instream, advanced to the beginning of the data
        return header, instream

    input_field_names = ['chrom1', 'pos51', 'pos31', 'strand1', 'chrom2', 'pos52', 'pos32', 'strand2', 'walk_pair_type']
    input_field_dtypes = {
        'chrom1': str,
        'pos51': np.int64,
        'pos31': np.int64,
        'strand1': str,
        'chrom2': str,
        'pos52': np.int64,
        'pos32': np.int64,
        'strand2': str,
        'walk_pair_type': str
    }
    input_field_numbers = {'chrom1': 1, 'pos51': 10, 'pos31': 12, 'strand1': 5,
                           'chrom2': 3, 'pos52': 11, 'pos32': 13, 'strand2': 6, 'walk_pair_type': 9}

    if pairs_path == '-':
        f_in = sys.stdin
        _, f_in = get_header(f_in)
    elif int(_pandas_version[0]) > 0:
        try:
            f_in = get_handle(pairs_path, mode='r', compression='infer')[0]
        except TypeError:
            f_in = get_handle(pairs_path, mode='r', compression='infer').handle
        _, f_in = get_header(f_in)
    else:
        f_in = pairs_path

    pairs_df = pd.read_csv(
        f_in,
        sep='\t',
        header=None,
        usecols=[input_field_numbers[name] for name in input_field_names],
        names=[k for k, v in sorted(input_field_numbers.items(), key=lambda item: item[1])],
        dtype=input_field_dtypes,
        iterator=True,
        chunksize=chunksize
    )
    return pairs_df

def rev_coor(pairs_df):
    '''
    Switch the coordinates of the pairs with reversed strand, as interval tree only works on interval with start < end
    :param pairs_df: dataframe for pairs
    :return: pairs_df with switched coordinate
    '''
    r1_rev = (pairs_df['pos31'] - pairs_df['pos51'] < 0).any()
    r2_rev = (pairs_df['pos32'] - pairs_df['pos52'] < 0).any()
    if r1_rev or r2_rev:
        print("Detect genome coordinate start > end, try to correct it based on strand")
        if r1_rev:
            pairs_df.loc[pairs_df['strand1'] == '-', ['pos51', 'pos31']] = pairs_df.loc[
                pairs_df['strand1'] == '-', ['pos31', 'pos51']].values
        if r2_rev:
            pairs_df.loc[pairs_df['strand2'] == '-', ['pos52', 'pos32']] = pairs_df.loc[
                pairs_df['strand2'] == '-', ['pos32', 'pos52']].values
    # check input again to see if there is reversed coordinate exists
    r1_rev = (pairs_df['pos31'] - pairs_df['pos51'] < 0).any()
    r2_rev = (pairs_df['pos32'] - pairs_df['pos52'] < 0).any()
    if r1_rev or r2_rev:
        print("Still detect genome coordinate start > end, skip these")
        pairs_df = pairs_df[
            ~((pairs_df['pos31'] - pairs_df['pos51'] < 0) | (pairs_df['pos32'] - pairs_df['pos52'] < 0))]
    return pairs_df

class BinpairPr(object):
    def __init__(self, nuc_df, pairs_df, ref_dict, olp_cutoff = 1, match_mode = 'best', rescue = False,
                 onlyintrachr = True, outdir = os.getcwd(),  numproc = 1):
        self.nuc_df = nuc_df
        self.pairs_df = pairs_df
        self.ref_dict = ref_dict
        self.numproc = numproc
        self.olp_cutoff = olp_cutoff
        self.match_mode = match_mode
        self.rescue = rescue
        self.intrachr = onlyintrachr
        self.outdir = outdir

    def _overlap(self, min1, max1, min2, max2):
        return max(0, min(max1, max2) - max(min1, min2))

    def _olp_filter(self, pairs_nuc):
        # return True/False pd.Series
        df = pairs_nuc.copy()
        if self.olp_cutoff == 1:
            return df
        elif isinstance(self.olp_cutoff, int) and self.olp_cutoff != 1:
            lcolp = (df['r1_olp_len'] < self.olp_cutoff) | (df['r2_olp_len'] < self.olp_cutoff)
        elif isinstance(self.olp_cutoff, str):
            # auto mode, from the exp, r1 and r2 almost have identical distribution, to save time, only use r1 data
            # use 3*MAD(median absolute deviation)/10
            print('olp_filter',len(df))
            r1_mad = np.median(np.absolute(df['r1_olp_len'] - np.median(df['r1_olp_len'])))
            self.olp_cutoff = 0.3 * r1_mad
            print("Auto mode activated, the overlapped cutoff is {}".format(self.olp_cutoff))
            lcolp = (df['r1_olp_len'] < self.olp_cutoff) | (df['r2_olp_len'] < self.olp_cutoff)
        else:
            print("Cannot recognize the olp_cut_mark, return all")
            lcolp = pd.Series([False] * len(df))
        return df[~lcolp]

    def _match_filter(self,olp_df):
        '''

        :param olp_df:
        :param match_mode:
        :return:
        '''
        if self.match_mode == 'multi':
            final_df = olp_df
        elif self.match_mode == 'best':
            olp_df['olp.sum'] = olp_df['r1_olp_len'] + olp_df['r2_olp_len']
            # https://stackoverflow.com/questions/12497402/remove-duplicates-by-columns-a-keeping-the-row-with-the-highest-value-in-column/68876659#68876659
            final_df = olp_df.sort_values(['olp.sum', 'pair_id'], ascending=False).drop_duplicates(
                'pair_id').sort_index()
            # final_df = olp_df.loc[olp_df.groupby('pair_id')['olp.sum'].idxmax()].sort_index()
        else:
            print("match mode not supported, switch to multi mode")
            final_df = olp_df
        return final_df

    def statsfigs(self, olp_len_all_df, prefix):
        print("Plotting the distribution of the length of the nucleosomes")
        plt.figure()
        df = self.nuc_df.copy()
        df['bin_len'] = df['End'] - df['Start']
        sns.displot(data=df, x='bin_len', kind='kde', bw_adjust=10, log_scale=2)
        plt.xlabel('Length (bp)')
        plt.savefig('{}/{}_bin_kde.png'.format(self.outdir, prefix))
        plt.close()
        print("Plotting the distribution of the length of the overlap region")
        try:
            olp_len_df_all_melt = pd.melt(olp_len_all_df, id_vars=['pair_id'], value_vars=['r1_olp_len', 'r2_olp_len'])
            plt.figure()
            sns.displot(data=olp_len_df_all_melt, x='value', hue='variable', kind='kde', log_scale=2)
            plt.axvline(x=self.olp_cutoff, color='red')
            plt.xlabel('Length (bp)')
            plt.savefig('{}/{}_olplen.png'.format(self.outdir, prefix))
            plt.close()
        except KeyError:
            print("DataFrame should at least contain pair_id, r1_olp_len and r2_olp_len column")
            exit(1)

    def _pairsintersect(self):
        # Initiate pyrange object
        nuc_df_pr = pr.PyRanges(self.nuc_df)
        pairs_r1_pr = pr.from_dict({"Chromosome": self.pairs_df['chrom1'],
                                    "Start": self.pairs_df['pos51'],
                                    "End": self.pairs_df['pos31'],
                                    "pair_id": self.pairs_df['pair_id']})
        pairs_r2_pr = pr.from_dict({"Chromosome": self.pairs_df['chrom2'],
                                    "Start": self.pairs_df['pos52'],
                                    "End": self.pairs_df['pos32'],
                                    "pair_id": self.pairs_df['pair_id']})
        # intersect
        if self.numproc == 1:
            pairs_r1_nuc = pairs_r1_pr.join(nuc_df_pr, report_overlap=True).as_df()
            pairs_r2_nuc = pairs_r2_pr.join(nuc_df_pr, report_overlap=True).as_df()
        else:
            try:
                pairs_r1_nuc = pairs_r1_pr.join(nuc_df_pr, report_overlap=True, nb_cpu=self.numproc).as_df()
                pairs_r2_nuc = pairs_r2_pr.join(nuc_df_pr, report_overlap=True, nb_cpu=self.numproc).as_df()
            except ModuleNotFoundError:
                print("Ray package is not found, which is required for paralleling compute, switch to single processor")
                pairs_r1_nuc = pairs_r1_pr.join(nuc_df_pr, report_overlap=True).as_df()
                pairs_r2_nuc = pairs_r2_pr.join(nuc_df_pr, report_overlap=True).as_df()
        # get all paired nucleosome combination
        pairs_nuc = pairs_r1_nuc.merge(pairs_r2_nuc, how='inner', on='pair_id')
        return pairs_nuc

    def coo_process(self):
        print("Total number of pairs: {}".format(len(self.pairs_df)))
        # find intersections between pairs and nucleosome
        pairs_nuc = self._pairsintersect()
        # only counting intra-chromosome pairs
        if self.intrachr:
            pairs_nuc = pairs_nuc[pairs_nuc['Chromosome_x'].astype('object') == pairs_nuc['Chromosome_y'].astype('object')]
        pairs_nuc = pairs_nuc[['pair_id', 'bin_id_x', 'Overlap_x', 'bin_id_y', 'Overlap_y']]
        pairs_nuc.columns = ['pair_id', 'bin_id_r1', 'r1_olp_len', 'bin_id_r2', 'r2_olp_len']
        # remove pairs with overlapped length less than cutoff
        pairs_nuc_filt = self._olp_filter(pairs_nuc)
        # dropped pairs
        if self.rescue:
            print("Rescue nolp pairs")
            nolp_pairs = self.pairs_df[~self.pairs_df['pair_id'].isin(set(pairs_nuc_filt['pair_id']))]
        else:
            nolp_pairs = pd.DataFrame()
        if self.intrachr:
            print("Total number of intra-chr pairs overlapped with nucleosome: {}".format(len(set(
                pairs_nuc_filt['pair_id']))))
        else:
            print("Total number of pairs overlapped with nucleosome: {}".format(len(set(
                pairs_nuc_filt['pair_id']))))
        # filter by multi-match or best-match mode
        pairs_nuc_filt_matched = self._match_filter(pairs_nuc_filt)
        print(
            "Total frequency of pairs overlapped with nucleosome: {}".format(len(pairs_nuc_filt_matched)))
        return nolp_pairs, pairs_nuc_filt_matched

    def coo_build(self, pairs_nuc_filt_matched):
        print("Transferring to upper triangular matrix")
        trans_idx = pairs_nuc_filt_matched['bin_id_r1'] > pairs_nuc_filt_matched['bin_id_r2']
        pairs_nuc_filt_matched.loc[trans_idx, ['bin_id_r1', 'bin_id_r2']] = \
            pairs_nuc_filt_matched.loc[trans_idx, ['bin_id_r2', 'bin_id_r1']].values
        print("Finalizing COO matrix")
        pairs_nuc_filt_matched = pairs_nuc_filt_matched.sort_values('pair_id')
        bin_df = pairs_nuc_filt_matched[['bin_id_r1', 'bin_id_r2']].groupby(['bin_id_r1', 'bin_id_r2'],
                                                                            as_index=False, sort=False).size()
        return bin_df

def nucload_save_cooler(cooler_path,bins_df,all_pixels_df):
    cooler.create_cooler(cool_uri=cooler_path,
                         bins=bins_df,
                         pixels=all_pixels_df,
                         ordered=True,
                         dtypes={'count': np.float32})
    return None






