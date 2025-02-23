#!/usr/bin/env python
#--coding:utf-8 --
import os
import time
import pyranges as pr
import numpy as np
from collections import defaultdict
import pandas as pd
import cooler
from tabulate import tabulate

def which(program):
    """
    Check if program exists on path. Adapted from MISO.
    """
    def is_exe(fpath):
        if not os.path.isfile(fpath):
            return False
        elif not os.access(fpath, os.X_OK):
            # If the file exists but is not executable, warn
            # the user
            print("WARNING: Found %s but it is not executable." %(fpath))
            print("Please ensure %s is executable." %(fpath))
            print("On Unix, use something like: ")
            print("  chmod +x %s" %(fpath))
            time.sleep(10)
            return False
        return True
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


# v3 add bi-direction function
class GenomicAnnot(object):
    def __init__(self, df, gene_annot_df, genomic_region_ranges=None, genomic_regions_idx=None, bidirect=False):
        """

        :param df: dataframe with first three columns: chr/start/end
        :param gene_annot_df: dataframe for gene reference
               gene_annot_df.columns = ['Chromosome', 'Start', 'End', 'Strand', 'GeneName', 'Count']
        :param genomic_region_ranges:
        :param genomic_regions_idx:
        """
        tmp_df = df.iloc[:, :3]
        tmp_df.columns = ['Chromosome', 'Start', 'End']
        self.df = tmp_df
        self.gene_annot_df = gene_annot_df
        if genomic_region_ranges is None:
            self.genomic_regions_ranges = {
                # prom:-1000 ~ 1000 of TTS
                'Promoter': [-1000, -1000, 1000, 1000],
                # 1000 of TTS to -1000 of tts
                'Genebody': [1000, -1000],
                'TTS': [-1000, -1000, 1000, 1000],
                'Proximal': [-10000, 1000, -1000, 10000],
                'Distal': [-250000, 10000, -10000, 250000]
            }
        else:
            self.genomic_regions_ranges = genomic_region_ranges
        if genomic_regions_idx is None:
            # genomic location: promoter > genebody > TTS > proximal> diatal > desserts(other)
            self.genomic_regions_idx = {'Promoter': 2 ** 5, 'TTS': 2 ** 4, 'Genebody': 2 ** 3, 'Proximal': 2 ** 2,
                                        'Distal': 2 ** 1}
        else:
            self.genomic_regions_idx = genomic_regions_idx
        self.bidirect = bidirect
    def _GenerateGenomicRegion(self):
        genomic_regions = {}
        for genoloc in self.genomic_regions_ranges:
            gene_annot_cp = self.gene_annot_df.copy()
            current_params = self.genomic_regions_ranges[genoloc]
            if genoloc not in ['Genebody', 'TTS']:
                gene_annot_cp[f'{genoloc}_start'] = np.where(gene_annot_cp['Strand'] == '+',
                                                             gene_annot_cp['Start'] + current_params[0],
                                                             gene_annot_cp['End'] + current_params[1])
                gene_annot_cp[f'{genoloc}_end'] = np.where(gene_annot_cp['Strand'] == '+',
                                                           gene_annot_cp['Start'] + current_params[2],
                                                           gene_annot_cp['End'] + current_params[3])
            elif genoloc == 'TTS':
                gene_annot_cp[f'{genoloc}_start'] = np.where(gene_annot_cp['Strand'] == '+',
                                                             gene_annot_cp['End'] + current_params[0],
                                                             gene_annot_cp['Start'] + current_params[1])
                gene_annot_cp[f'{genoloc}_end'] = np.where(gene_annot_cp['Strand'] == '+',
                                                           gene_annot_cp['End'] + current_params[2],
                                                           gene_annot_cp['Start'] + current_params[3])
            elif genoloc == 'Genebody':
                gene_annot_cp[f'{genoloc}_start'] = gene_annot_cp['Start'] + current_params[0]
                gene_annot_cp[f'{genoloc}_end'] = gene_annot_cp['End'] + current_params[1]
            else:
                print('Not possible condition')
            gene_annot_cp.loc[gene_annot_cp[f'{genoloc}_start'] < 0, f'{genoloc}_start'] = 1
            gene_annot_cp.loc[gene_annot_cp[f'{genoloc}_end'] < 0, f'{genoloc}_end'] = 1
            gene_annot_cp = gene_annot_cp[['Chromosome', f'{genoloc}_start', f'{genoloc}_end', 'GeneName']]
            gene_annot_cp.columns = ['Chromosome', 'Start', 'End', 'GeneName']
            # bidirect
            if self.bidirect:
                if genoloc not in ['Genebody', 'TTS', 'Promoter']:
                    gene_annot_cp_bi = self.gene_annot_df.copy()
                    current_params = self.genomic_regions_ranges[genoloc]
                    gene_annot_cp_bi[f'{genoloc}_start'] = np.where(gene_annot_cp_bi['Strand'] == '+',
                                                                    gene_annot_cp_bi['Start'] - current_params[2],
                                                                    gene_annot_cp_bi['End'] - current_params[3])
                    gene_annot_cp_bi[f'{genoloc}_end'] = np.where(gene_annot_cp_bi['Strand'] == '+',
                                                                  gene_annot_cp_bi['Start'] - current_params[0],
                                                                  gene_annot_cp_bi['End'] - current_params[1])
                    gene_annot_cp_bi.loc[gene_annot_cp_bi[f'{genoloc}_start'] < 0, f'{genoloc}_start'] = 1
                    gene_annot_cp_bi.loc[gene_annot_cp_bi[f'{genoloc}_end'] < 0, f'{genoloc}_end'] = 1
                    gene_annot_cp_bi = gene_annot_cp_bi[['Chromosome', f'{genoloc}_start', f'{genoloc}_end', 'GeneName']]
                    gene_annot_cp_bi.columns = ['Chromosome', 'Start', 'End', 'GeneName']
                    gene_annot_cp_bi['GeneName'] = gene_annot_cp_bi['GeneName'] + '_bi'
                else:
                    gene_annot_cp_bi = pd.DataFrame(columns=['Chromosome', 'Start', 'End', 'GeneName'])
                gene_annot_cp_final = pd.concat([gene_annot_cp,gene_annot_cp_bi])
            else:
                gene_annot_cp_final = gene_annot_cp
            genomic_regions[genoloc] = gene_annot_cp_final
        return genomic_regions
    def _genoloc_annot_dict(self, regions_pr, gene_region_pr):
        regions_annot = regions_pr.join(gene_region_pr).as_df()
        # in case no overlaps found
        if len(regions_annot) == 0:
            return {}
        else:
            regions_set = set(
                regions_annot[['Chromosome', 'Start', 'End', 'GeneName']].itertuples(index=False, name=None))
            regions_annot_dict = defaultdict(list)
            for x in regions_set:
                regions_annot_dict['_'.join([x[0], str(x[1]), str(x[2])])].append(x[3])
            regions_annot_dict = {key: ','.join(value) for key, value in regions_annot_dict.items()}
        return regions_annot_dict
    def annot(self):
        df_annot = self.df.copy()
        df_annot['key'] = df_annot['Chromosome'].astype(str) + '_' + df_annot['Start'].astype(str) + '_' + df_annot[
            'End'].astype(str)
        print("Generating genomic region range from reference")
        genomic_regions = self._GenerateGenomicRegion()
        print("Annotating")
        df_annot_pr = pr.PyRanges(df_annot)
        for genoloc in genomic_regions:
            genomic_regions_pr = pr.PyRanges(genomic_regions[genoloc])
            genolocdict = self._genoloc_annot_dict(df_annot_pr, genomic_regions_pr)
            df_annot[f'{genoloc}'] = df_annot['key'].map(genolocdict)
            df_annot[f'{genoloc}_idx'] = df_annot[f'{genoloc}'].notnull().astype('int') * self.genomic_regions_idx[
                genoloc]
        idx_cols = [f'{genoloc}_idx' for genoloc in genomic_regions]
        df_annot['Desert'] = 'No Gene'
        df_annot['sum'] = df_annot[idx_cols].sum(axis=1)
        # initial final_annot, order is important here
        df_annot['genomeLoc_annot'] = 'Desert'
        df_annot.loc[df_annot['sum'] >= self.genomic_regions_idx['Distal'], 'genomeLoc_annot'] = 'Distal'
        df_annot.loc[df_annot['sum'] >= self.genomic_regions_idx['Proximal'], 'genomeLoc_annot'] = 'Proximal'
        df_annot.loc[df_annot['sum'] >= self.genomic_regions_idx['Genebody'], 'genomeLoc_annot'] = 'Genebody'
        df_annot.loc[df_annot['sum'] >= self.genomic_regions_idx['TTS'], 'genomeLoc_annot'] = 'TTS'
        df_annot.loc[df_annot['sum'] >= self.genomic_regions_idx['Promoter'], 'genomeLoc_annot'] = 'Promoter'
        # Apply the function to create the new column 'f'
        df_annot['genes'] = df_annot.apply(lambda x: x[x['genomeLoc_annot']], axis=1)
        # df_annot.loc[df_annot['genomeLoc_annot']=='Desert','genes'] = 'Not Avail'
        gene_df_annot_final = df_annot[['Chromosome', 'Start', 'End', 'genomeLoc_annot', 'genes']]
        return gene_df_annot_final

# add an extra col to an existed .cool
def addCols(clr,store_name,store_data):
    with clr.open("r+") as grp:
        if store_name in grp["bins"]:
            del grp["bins"][store_name]
        h5opts = {"compression": "gzip", "compression_opts": 6}
        grp["bins"].create_dataset(store_name, data=store_data, **h5opts)

def get_qc(stat_f, outf):
    # from dovetail get_qc.py
    output_dict = {}
    with open(stat_f, 'r') as f:
        for line in f:
            attrs = line.split()
            output_dict[attrs[0]] = attrs[1]
    data = []
    total_reads = int(output_dict["total"])
    total_reads_str = format(total_reads, ",d")
    data.append(["Total Read Pairs", total_reads_str, "100%"])
    unmapped_reads = int(output_dict["total_unmapped"])
    percent_unmapped = round(unmapped_reads * 100.0 / total_reads, 2)
    unmapped_reads_str = format(unmapped_reads, ",d")
    data.append(["Unmapped Read Pairs", unmapped_reads_str, f"{percent_unmapped}%"])
    mapped_reads = int(output_dict["total_mapped"])
    percent_mapped = round(mapped_reads * 100.0 / total_reads, 2)
    mapped_reads_str = format(mapped_reads, ",d")
    data.append(["Mapped Read Pairs", mapped_reads_str, f"{percent_mapped}%"])
    dup_reads = int(output_dict["total_dups"])
    percent_dups = round(dup_reads * 100.0 / total_reads, 2)
    dup_reads_str = format(dup_reads, ",d")
    data.append(["PCR Dup Read Pairs", dup_reads_str, f"{percent_dups}%"])
    nodup_reads = int(output_dict["total_nodups"])
    percent_nodups = round(nodup_reads * 100.0 / total_reads, 2)
    nodup_reads_str = format(nodup_reads, ",d")
    data.append(["No-Dup Read Pairs", nodup_reads_str, f"{percent_nodups}%"])
    cis_reads = int(output_dict["cis"])
    percent_cis = round(cis_reads * 100.0 / nodup_reads, 2)
    cis_reads_str = format(cis_reads, ",d")
    data.append(["No-Dup Cis Read Pairs", cis_reads_str, f"{percent_cis}%"])
    trans_reads = int(output_dict["trans"])
    percent_trans = round(trans_reads * 100.0 / nodup_reads, 2)
    trans_reads_str = format(trans_reads, ",d")
    data.append(["No-Dup Trans Read Pairs", trans_reads_str, f"{percent_trans}%"])
    cis_gt1kb = int(output_dict["cis_1kb+"])
    cis_lt1kb = cis_reads - cis_gt1kb
    percent_cis_lt1kb = round(cis_lt1kb * 100.0 / nodup_reads, 2)
    percent_cis_gt1kb = round(cis_gt1kb * 100.0 / nodup_reads, 2)
    cis_gt1kb_str = format(cis_gt1kb, ",d")
    cis_lt1kb_str = format(cis_lt1kb, ",d")
    valid_read_pairs = int(output_dict["trans"]) + int(output_dict["cis_1kb+"])
    percent_valid_read_pairs = round(valid_read_pairs * 100.0 / nodup_reads, 2)
    valid_read_pairs_str = format(valid_read_pairs, ",d")
    data.append(
        ["No-Dup Valid Read Pairs (cis >= 1kb + trans)", valid_read_pairs_str, f"{percent_valid_read_pairs}%"])
    data.append(["No-Dup Cis Read Pairs < 1kb", cis_lt1kb_str, f"{percent_cis_lt1kb}%"])
    data.append(["No-Dup Cis Read Pairs >= 1kb", cis_gt1kb_str, f"{percent_cis_gt1kb}%"])
    cis_gt10kb = int(output_dict["cis_10kb+"])
    percent_cis_gt10kb = round(cis_gt10kb * 100.0 / nodup_reads, 2)
    cis_gt10kb_str = format(cis_gt10kb, ",d")
    data.append(["No-Dup Cis Read Pairs >= 10kb", cis_gt10kb_str, f"{percent_cis_gt10kb}%"])
    # Convert to DataFrame and save to file
    df = pd.DataFrame(data, columns=["Metric", "Count", "Percentage"])
    df.to_csv(outf, index=False)