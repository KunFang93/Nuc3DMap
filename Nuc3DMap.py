#!/usr/bin/env python
#--coding:utf-8 --


#################################################################################################
#################################################################################################
########                                                                                 ########
########    Nucleosome-level 3D genome organization Map                                  ########
########                                                                                 ########
########    Author:  Kun Fang                                                            ########
########                                                                                 ########
########                                                                                 ########
########    Working Environment:  Python3                                                ########
########                                                                                 ########
########    Date:      2024-04-26                                                        ########
########                                                                                 ########
########                                                                                 ########
#################################################################################################
#################################################################################################

import os, re
import time, datetime
import click
import psutil
import resource
import cooler
import Nuc3DMap_references
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import pickle
from datetime import date
from Nuc3DMap_utilities import which
from Nuc3DMap_nucprep import prepfromfastq, prepfrombam, nuc_calling, alignfq, remove_extension_and_suffix, withinRegularChrom
from Nuc3DMap_nucload import PrepNuc, load_raw_pairs, rev_coor, BinpairPr, nucload_save_cooler
from Nuc3DMap_nucmerge import chrom_norm, iMHiC, visualConverge, nucmerge_save_cooler
from Nuc3DMap_nucTD import preTADclr, NucDom, callTADs, makeConsecutiveCoor, NucDom_RG, TransferLearning
from Nuc3DMap_nucIL import build_RGMap, NucIL_detection, transfer_NucLoop

# set memory limitation
def set_memory_limit(limit):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
def set_memory_limit(percentage):
    # Get total memory
    total_memory = psutil.virtual_memory().total
    # Calculate the limit (80% of total memory)
    limit = int(total_memory * percentage)
    # Set the soft limit
    resource.setrlimit(resource.RLIMIT_AS, (limit, resource.RLIM_INFINITY))

def is_compressed_file(filename):
    compressed_extensions = ('.zip', '.gz', '.bz2', '.xz')
    return filename.endswith(compressed_extensions)

# Set the memory limit to 80% of the total memory
set_memory_limit(0.8)

# class for setting up subcommand order.
# Inspired by https://stackoverflow.com/questions/47972638/how-can-i-define-the-order-of-click-sub-commands-in-help
class SpecialHelpOrder(click.Group):

    def __init__(self, *args, **kwargs):
        self.help_priorities = {}
        super(SpecialHelpOrder, self).__init__(*args, **kwargs)

    def get_help(self, ctx):
        self.list_commands = self.list_commands_for_help
        return super(SpecialHelpOrder, self).get_help(ctx)

    def list_commands_for_help(self, ctx):
        """reorder the list of commands when listing the help"""
        commands = super(SpecialHelpOrder, self).list_commands(ctx)
        return (c[1] for c in sorted(
            (self.help_priorities.get(command, 1), command)
            for command in commands))

    def command(self, *args, **kwargs):
        """Behaves the same as `click.Group.command()` except capture
        a priority for listing command names in help.
        """
        help_priority = kwargs.pop('help_priority', 1)
        help_priorities = self.help_priorities

        def decorator(f):
            cmd = super(SpecialHelpOrder, self).command(*args, **kwargs)(f)
            help_priorities[cmd.name] = help_priority
            return cmd

        return decorator

required_programs = ["bwa","pairtools","deeptools","samtools","bedtools"]
# "macs2","epic2","fastqc","trim_galore","bowtie","bowtie2" are needed if use nuchmm-prep
for prog in required_programs:
    p = which(prog)
    # print("Checking if %s is available" %(prog))
    if p is None:
        print(" - Cannot find %s! Please install it!" %(prog))
        exit(1)
    else:
        continue

class Config(object):

    def __init__(self):
        self.verbose = False

pass_config = click.make_pass_decorator(Config, ensure=True)
@click.group(cls=SpecialHelpOrder)
@click.version_option(version='0.1')
@pass_config
def cli(config):
    pass

error_message = "Please contact author kfang@mcw.edu for further supporting, thank you!"

@cli.command(help_priority=1, help='Convert .fq/bam into .pairs and nuc_loc.bed')
@click.option('--inputfiles', '-ifs', type=click.Path(exists=True),
              help='Inputfiles, each row contains either fastq or bm for a sample. Please refer to example_input check details')
@click.option('--gsize', '-gs', type=click.Path(exists=True), help = 'referencec genome size file, e.g, hg38.chrom.sizes')
@click.option('--prefixname', '-name', type=str, help = 'Specify the prefix name')
@click.option('--minmapq', '-mq', default=30, type=int, help = 'minimum mapq for quality control')
@click.option('--bwafasta', '-bfa', type=click.Path(), help = 'Required if input format is fastq, the path for refernce fasta (bwa <idxbase>)')
@click.option('--inpsdir','-inps', type=click.Path(), help = 'Required if input has MNase-seq, the path of iNPS.py. '
                                                             'For example, /data/NucHMM/scripts/iNPS_V1.2.2.py.')
@click.option('--tmpdir', '-td', type=click.Path(), help = 'temporary folder to store intermediate results')
@click.option('--threads', '-p', default=5, type=int, help = 'Number of threads, default 5')
def nucprep(inputfiles, gsize, prefixname, minmapq, bwafasta, inpsdir, tmpdir, threads):
    # set tmpdir
    if tmpdir is None:
        tmpdir = os.path.abspath('./tmpDir')
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    else:
        tmpdir = os.path.abspath(tmpdir)

    # check if all required file are inputted correctly
    exit_flag = False
    with open(inputfiles, 'r') as inputf:
        for line in inputf:
            if line.startswith("#"):
                continue
            line = line.strip()
            line_info = re.split('[\t| ]', line)
            line_type = line_info[-1]
            line_fs = line_info[:-1]
            line_prefix = Path(line_fs[0]).stem
            if line_fs[0].endswith(('.fq', '.fastq', '.fq.gz', '.fastq.gz')):
                # check required file
                if bwafasta is None:
                    print("Process Fastq file require idxbase file for bwa, ues -bfa to input")
                    exit_flag = True
                else:
                    bwafasta = os.path.abspath(bwafasta)
            if line_type in ['MNase', 'mnase', 'MNASE']:
                if inpsdir is None:
                    print("Process MNase file require iNPS.py file for nucleosome calling, ues -ipns to input the absolute path")
                    exit_flag = True
                else:
                    inpsdir = os.path.abspath(inpsdir)
    if exit_flag:
        print('Please input required files')
        exit(1)

    # process each sample
    with open(inputfiles, 'r') as inputf:
        for line in inputf:
            if line.startswith("#"):
                continue
            line = line.strip()
            line_info = re.split('[\t| ]', line)
            line_type = line_info[-1]
            line_fs = line_info[:-1]
            line_prefix = remove_extension_and_suffix(line_fs[0])
            # set prefix
            if prefixname is None:
                prefix = line_prefix
            else:
                prefix = prefixname
            if line_type in ['Hi-C', 'Micro-C', 'HiC', 'MicroC', 'hic', 'microc']:   # generate pairs from Hi-C or Micro-C data
                print(f"Processing {line_type}")
                if line_fs[0].endswith(('.fq', '.fastq', '.fq.gz', '.fastq.gz')):
                    prepfromfastq(fastq_r1=os.path.abspath(line_fs[0]), fastq_r2=os.path.abspath(line_fs[1]), prefix=f"{prefix}_{line_type}", ref_fasta=bwafasta, gsize=gsize, minmapq=minmapq, tmpdir=tmpdir,  nthreads=threads)
                elif line_fs[0].endswith('.bam'):
                    prepfrombam(bam_f=os.path.abspath(line_fs[0]), prefix=f"{prefix}_{line_type}", gsize=gsize, minmapq=minmapq, tmpdir=tmpdir, nthreads=threads)
                else:
                    print(f"Unrecognized input format, Nuc3DMap nucprep only supports fastq and bam input."
                          f"{error_message}")
                    exit(1)
            elif line_type in ['MNase', 'mnase', 'MNASE']:    # generate nucleosome locations from MNase data
                print(f"Processing {line_type}")
                # process chroms.filt.bed to ensure only regular chroms
                withinRegularChrom(bwafasta, tmpdir)
                if line_fs[0].endswith(('.fq', '.fastq', '.fq.gz', '.fastq.gz')):
                    if line_info[-2] == 'SE':
                        seqtype = alignfq(line_fs, 'SE', f'{tmpdir}/{prefix}_{line_type}', minmapq, bwafasta, tmpdir, threads)
                    elif line_info[-2] == 'PE':
                        seqtype = alignfq(line_fs, 'PE', f'{tmpdir}/{prefix}_{line_type}', minmapq, bwafasta, tmpdir, threads)
                    else:
                        print("seqtype only limited to SE/PE")
                        exit(1)
                    nuc_calling(f'{tmpdir}/{prefix}_{line_type}.bam', seqtype, inpsdir, tmpdir)
                elif line_fs[0].endswith('.bam'):
                    nuc_calling(os.path.abspath(line_fs[0]), line_fs[1], inpsdir, tmpdir)
                else:
                    print(f"Unrecognized input format, Nuc3DMap nucprep only supports fastq and bam input."
                          f"{error_message}")
                    exit(1)
            else:
                print(f"Unrecognized input type, Nuc3DMap nucprep only supports Hi-C, Micro-C and MNase."
                      f"{error_message}")
                exit(1)

@cli.command(help_priority=2, help='Build nucleosome-level contact map')
@click.option('--pairsfile', '-pf',type=click.Path(exists=True), required = True, help='Input pairs')
@click.option('--nuclocfile', '-nucf', type=click.Path(exists=True), required=True, help='The nucleosome position files')
@click.option('--geneannot', '-gt', type=click.Path(exists=True), required=True, help='reference genes annotation file')
@click.option('--outdir', '-od', type=click.Path(), help='Specify the output files path and name.')
@click.option('--refgenome', '-refg', default='hg38', help='The name of reference genome, currently accept hg19|hg38')
@click.option('--reportnooverlap', '-rnolp', is_flag=True, help='Flag of saving pairs that are not directed overlapped or overlapped length less than cutoff')
@click.option('--statsreport', '-sr', is_flag=True, help='Flag of plotting distributions of variables in nucload')
@click.option('--overlapcut','-olpc', default = 1, help='cutoff for overlapping length between pairs and map bins, default 1 (auto mode is -1)')
@click.option('--match', '-m', default='best', help='best: one pair only belong to one bin | multi: one pair belong can assgin to multiple bins')
@click.option('--chunksize', '-cs', default = 15000000, type=int, help='The chunksize to load the .pairs file')
@click.option('--sampling', '-spl', default = 1.0, type=float, help='Sampling pairs for robust test, ignore for normal usage, [0,1]')
@click.option('--nproc', '-np', default = 1, type=int, help='The number of parallel processors')
def nucload(pairsfile, nuclocfile, geneannot, outdir, refgenome, reportnooverlap, statsreport, overlapcut, match,
            chunksize, sampling, nproc):
    prefix = Path(pairsfile).stem
    if outdir is None:
        outdir = './'

    if refgenome == 'hg38':
        ref_dict = Nuc3DMap_references.hg38
    elif refgenome == 'hg19':
        ref_dict = Nuc3DMap_references.hg19
    else:
        # ToDo: add option that can input customed reference dictionary
        print("reference not stored, please contact kfang@mcw.edu for the support")
        exit(1)

    # prepare nuc-interval bin
    # hyperparameters
    nuc_extsize = 35  # extend from iNPS result as it calls the ~80 bp nucleosome core region
    start = time.time()
    if is_compressed_file(nuclocfile):
        print("Pairs in zipped format, unzip it")
        subprocess.call(f"gunzip {nuclocfile}",shell=True)
        nuclocfile = f'{str(Path(nuclocfile).parent)}/{Path(nuclocfile).stem}'
        # check exist
        if os.path.exists(nuclocfile):
            print("Unzip successfully")
        else:
            print(f"Failed to unizp {nuclocfile}, please try unzip manually and re-run the command")
    prenuc = PrepNuc(nuc_f=nuclocfile, gene_f=geneannot, ref_dict=ref_dict,
                     prefix=prefix, outdir=outdir, extsize=nuc_extsize, min_interval=10)
    nucbins = prenuc.binproc()

    if overlapcut == -1:
        overlapcut == 'auto'
    # load pairs
    pairs_df = load_raw_pairs(pairsfile, chunksize)
    tmp_coo_list = []
    for i, pairs_chunk in enumerate(pairs_df):
        print('Processing Chunk{}'.format(i))
        pairs_chunk = rev_coor(pairs_chunk)
        pairs_chunk['pair_id'] = pairs_chunk.index.values
        # sampling
        if sampling == 1:
            pairs_chunk = pairs_chunk.copy()
        else:
            pairs_chunk = pairs_chunk.sample(frac=sampling, random_state=34)
        # coo built
        binpairpr = BinpairPr(nucbins[['Chromosome','Start','End','bin_id']],
                              pairs_chunk, ref_dict, olp_cutoff = overlapcut, match_mode = match,
                              rescue = reportnooverlap, numproc = nproc)
        nolp_pairs, binpair_inter = binpairpr.coo_process()
        coo_mat = binpairpr.coo_build(binpair_inter)
        tmp_coo_list.append(coo_mat)

        if reportnooverlap:
            print("Saving pairs that are not directed overlapped or overlapped length less than cutoff")
            # Set writing mode to append after first chunk
            mode = 'w' if i == 0 else 'a'

            # Add header if it is the first chunk
            header = i == 0

            nolp_pairs.to_csv(
                "{}/{}.nolp.pairs".format(outdir, prefix),
                sep='\t',
                index=False,  # Skip index column
                header=header,
                mode=mode)

        if statsreport and i == 0:
            binpairpr.statsfigs(binpair_inter, prefix)

    coo_df = pd.concat(tmp_coo_list).groupby(['bin_id_r1', 'bin_id_r2'], as_index=False).sum('size')
    coo_df[['bin_id_r1','bin_id_r2']] = coo_df[['bin_id_r1','bin_id_r2']].astype(int)

    # generate .cool file
    print("Generating .cool file")
    coo_df.columns = ['bin1_id', 'bin2_id', 'count']
    nucbins.columns = ['chrom', 'start', 'end', 'Itype', 'spacing', 'positioning','bin_id','nuc']
    # # save tmp results
    # coo_df.to_pickle('./coo_df_v2.pkl')
    # nucbins.to_pickle('./nucbins_v2.pkl')
    if os.path.exists(outdir):
        nucload_save_cooler(f'{outdir}/{prefix}.{sampling}.cool', nucbins, coo_df)
    elif outdir.split('.')[-1] == 'cool':
        nucload_save_cooler(f'{outdir}', nucbins, coo_df)
    else:
        print(f"Cannot recognize od parameter, save file to {os.getcwd()}")
        nucload_save_cooler(f'{os.getcwd()}/{prefix}.{sampling}.cool', nucbins, coo_df)
    end = time.time()
    # clear cache
    del(coo_df)
    del(nucbins)
    print("Processing Compeleted with {}".format(datetime.timedelta(seconds=end-start)))

@cli.command(help_priority=3, help='Merge Hi-C and Micro-C contact map')
@click.option('--microc', '-mc', type=click.Path(exists=True), required=True, help='MicroC.cool resulted from nucload.')
@click.option('--hic', '-hc', type=click.Path(exists=True), required=True, help='HiC.cool resulted from nucload.')
@click.option('--enzymedigestf', '-edf', type=click.Path(exists=True), required=True, help='enzyme digestion file')
@click.option('--outdir', '-od', type=click.Path(), help='Specify the output files path and name.')
@click.option('--refgenome', '-refg', default='hg38', help='The name of reference genome, currently accept hg19|hg38, default hg38')
@click.option('--visualize','-vis', is_flag=True, help='Visualize convergence and MD plot, which will slow down the process')
@click.option('--savetmpfile','-smf', is_flag=True, help='Save temporary files')
@click.option('--usetmpfile','-umf', is_flag=True, help='Skipp first step, directly use temporary files')
def nucmerge(microc, hic, enzymedigestf, outdir, refgenome, visualize, savetmpfile, usetmpfile):
    if refgenome == 'hg38':
        ref_dict = Nuc3DMap_references.hg38
    elif refgenome == 'hg19':
        ref_dict = Nuc3DMap_references.hg19
    else:
        # ToDo: add option that can input customed reference dictionary
        print("reference not stored, please contact kfang@mcw.edu for the support")
        exit(1)

    prefix = os.path.basename(microc).split('_')[0]
    if not usetmpfile:
        print("Processing Digestion file")
        digest_df = pd.read_table(enzymedigestf, header=None)
        digest_df.columns = ['Chromosome', 'Start', 'End', 'bin_id']
        digest_df_gb = digest_df.groupby('Chromosome')
        digest_chrs = {}
        for chr in ref_dict:
            digest_chr = digest_df_gb.get_group(chr)
            digest_chr = digest_chr.iloc[:-1, :3]
            digest_chr['Start'] = digest_chr['End'] - 1
            digest_chr['End'] += 1
            digest_chrs[chr] = digest_chr
        # load clr
        microc_clr = cooler.Cooler(microc)
        hic_clr = cooler.Cooler(hic)
        # generate chrom_iter
        res_dict, hic_coos, microc_coos, bins_list = chrom_norm(microc_clr, hic_clr, digest_chrs, ref_dict, prefix, outdir,
                                                                plotMD=visualize)
        # save to coolers
        if savetmpfile == True:
            nucmerge_save_cooler(f'{outdir}/{prefix}_HiC_KRFNormed.cool', bins_list, hic_coos, ref_dict)
            nucmerge_save_cooler(f'{outdir}/{prefix}_MicroC_KRFNormed.cool', bins_list, microc_coos, ref_dict)

        if visualize:
            with open(f'{outdir}/{prefix}_merge_res_dict.pickle', 'wb') as handle_res:
                pickle.dump(res_dict, handle_res, protocol=pickle.HIGHEST_PROTOCOL)
            visualConverge(ref_dict, res_dict, prefix, outdir)
    else:
        print(f"Use the exist {prefix}_HiC_KRFNormed.cool and {prefix}_MicroC_KRFNormed.cool")
        microc_clr = cooler.Cooler(f'{prefix}_MicroC_KRFNormed.cool')
        hic_clr = cooler.Cooler(f'{prefix}_HiC_KRFNormed.cool')
        chroms = microc_clr.chromnames
        microc_coos = [microc_clr.matrix(sparse=True, balance=False).fetch(chrom) for chrom in chroms]
        hic_coos = [hic_clr.matrix(sparse=True, balance=False).fetch(chrom) for chrom in chroms]
        bins_list = [microc_clr.bins().fetch(chrom) for chrom in chroms]
    # # integrate micro and hic
    print("Integration start")
    bins_list, imhic_coos = iMHiC(microc_coos, hic_coos, bins_list, ref_dict)
    todays_date = date.today()
    print("Writing to cool file")
    if os.path.exists(outdir):
        nucmerge_save_cooler(f'{outdir}/{prefix}_iMHiC_{todays_date.month}{todays_date.day}{todays_date.year}.cool',
                             bins_list,
                             imhic_coos, ref_dict)
    elif outdir.split('.')[-1] == 'cool':
        nucmerge_save_cooler(f'{outdir}',
                             bins_list,
                             imhic_coos, ref_dict)
    else:
        print(f"Cannot recognize od parameter, save file to {os.getcwd()}")
        nucmerge_save_cooler(
            f'{os.getcwd()}/{prefix}_iMHiC_{todays_date.month}{todays_date.day}{todays_date.year}.cool', bins_list,
            imhic_coos, ref_dict)


@cli.command(help_priority=4, help='Calling NucDom/Boundary/Gap')
@click.option('--imhiccool', '-imhc', type=click.Path(exists=True), required=True, help='Integrated MicroC-HiC.cool resulted from nucmerge.')
@click.option('--winsize', '-ws', default = 10, type=int, help='window size for calling NucDom')
@click.option('--outdir', '-od', type=click.Path(), help='Specify the output directory')
@click.option('--shiftsize', '-ss', default = 0, type=float, help='Empirical shift size for boundaries, default 0')
@click.option('--rgmap', '-rg', is_flag=True, help='Input is renormalized grouping map')
@click.option('--translearning','-tl', type=str,
              help='Usage: -tl H1_iMHiC.txt,H1_CTCF.bigwig,foo_CTCF.bigwig; Using H1 NucB (hg38) and H1 CTCF signal to infer NucB based on closed CTCF signal between H1 and targeted cell type\'s CTCF signal, '
                   'Used with cautions and only consider to use this parameter when sequence depth is not enough. Currently only support human hg38')
def nuctd(imhiccool, winsize, outdir, shiftsize, rgmap, translearning):
    # generate chrom_iter
    prefix = os.path.basename(imhiccool).split('_')[0]
    if outdir is None:
        outdir = './'
    imhic_clr = cooler.Cooler(imhiccool)
    chroms = imhic_clr.chromnames
    # hyperparameters
    if rgmap:
        tolerance = 1
        mingapsize = 3
    else:
        tolerance = 2
        mingapsize = 10
    winsize = winsize
    tads_list = []
    for chrom in chroms:
        print(f"Processing {chrom}")
        bins_chr, csr_mat_chr, gaps = preTADclr(imhic_clr, chrom, tolerance=tolerance, mingapsize=mingapsize, rgmap=rgmap)
        if rgmap:
            domains = NucDom_RG(bins_chr, csr_mat_chr, winsize)
        else:
            domains = NucDom(bins_chr, csr_mat_chr, winsize)
        result_df = callTADs(domains, gaps, bins_chr, rgmap=rgmap)
        tads_list.append(result_df)

    # write files
    tads_df = pd.concat(tads_list)
    # tads_df.to_pickle(f'{outdir}/tads_df.pkl')
    tads_df_adj = makeConsecutiveCoor(tads_df,chroms,shiftsize)
    domains_df = tads_df_adj[tads_df_adj['tag']=='domain']
    if shiftsize != 0:
        outname = f'{prefix}_iMHiC.win{winsize}.ss{shiftsize}'
    else:
        outname = f'{prefix}_iMHiC.win{winsize}'

    if translearning:
        h1_nucdom_f, h1_ctcf_f, targted_ctcf_f = translearning.split(',')
        tads_df_final = TransferLearning(tads_df_adj, h1_nucdom_f, h1_ctcf_f, imhic_clr, targted_ctcf_f)
        domains_df_final = tads_df_final[tads_df_final['tag'] == 'domain']
    else:
        tads_df_final = tads_df_adj.copy()
        domains_df_final = domains_df.copy()

    if os.path.exists(outdir):
        tads_df_final.to_csv(f'{outdir}/{outname}.txt', sep='\t', index=False)
        domains_df_final.to_csv(f'{outdir}/{outname}.domain.bed', sep='\t', index=False, header=None)
    else:
        print(f'{outdir} is not exist, save to current folder')
        tads_df_final.to_csv(f'./{outname}.txt', sep='\t', index=False)
        domains_df_final.to_csv(f'./{outname}.domain.bed', sep='\t', index=False, header=None)


@cli.command(help_priority=5, help='Detecting Nuc Interation Loci (NucIL)')
@click.option('--imhiccool', '-imhc', type=click.Path(exists=True), required=True, help='Integrated MicroC-HiC.cool resulted from nucmerge.')
@click.option('--tads', '-t', type=click.Path(exists=True), required=True, help='NucDom.txt from nucdom')
@click.option('--rgmap', '-rg', type=click.Path(exists=True), help='If RGMap have built, use this command to skip it')
@click.option('--distanceup', '-dup', default = 2000000, type=int, help='The maximum distance for detecting NucIL, default 2Mb')
@click.option('--distancelow', '-dlow', default = 5000, type=int, help='The maximum distance for detecting NucIL, default 5kb')
@click.option('--fdr', '-q', default = 0.1, type=float, help='FDR cutoff for detecting NucIL, default 0.1')
@click.option('--sigma', '-s', default = 0.5, type=float, help='sigma for initial Gaussian smoothing , default 0.5')
@click.option('--extmode', '-em', default=None, help='Choose among None|constant|adjacent, represents NoExt|Ext +/-3000|Ext adjacent Nucdom.')
@click.option('--outdir', '-od', type=click.Path(), help='Specify the output directory')
@click.option('--nucloop','-nl',is_flag=True, help="Output NucLoops that at least one end located in promoter region. -gt Reference bed file is requested")
@click.option('--geneannot','-gt', type=click.Path(),help="'reference genes annotation file'")
@click.option('--nproc', '-np', default = 8, type=int, help='The number of parallel processors')
def nucil(imhiccool, tads, rgmap, outdir, distanceup, distancelow, fdr, sigma, nproc, extmode, geneannot, nucloop):
    prefix = os.path.basename(imhiccool).split('_')[0]
    imhic_clr = cooler.Cooler(imhiccool)
    tads_df = pd.read_table(tads)
    # build RGMap
    if not rgmap:
        # tag id in RGMap: tag2id = {'gap': 0, 'boundary': 1, 'domain': 2}
        build_RGMap(imhic_clr,tads_df,f'{outdir}/{prefix}_iMHiC_RGMap.cool')
        rgmap = f'{outdir}/{prefix}_iMHiC_RGMap.cool'
    else:
        print(f"NucILoc Calling on existed {rgmap}")
        rgmap = rgmap
    blocksum_clr = cooler.Cooler(rgmap)
    # Detect NucIL
    nproc = min(os.cpu_count(),nproc)
    nucil_df, nucil_bedpe = NucIL_detection(blocksum_clr,
                                                  distance_up_filter = distanceup,
                                                  distance_low_filter = distancelow,
                                                  pt = fdr, nprocesser = nproc, sigma0=sigma,
                                                  extmode = extmode)
    nucil_df.to_csv(f'{outdir}/{prefix}_iMHiC_NucILoc.txt', index=False, sep='\t')
    nucil_bedpe.to_csv(f'{outdir}/{prefix}_iMHiC_NucILoc.bedpe', sep='\t', index=False, header=None)
    if nucloop:
        if geneannot is None:
            print("reference bed file for gene annotation is require, use -gt")
            exit(1)
        else:
            nucloops_df = transfer_NucLoop(nucil_df, geneannot)
            nucloops_df.to_csv(f'{outdir}/{prefix}_iMHiC_NucLoops.csv', index=False)
