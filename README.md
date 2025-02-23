# Nuc3DMap
A comprehensive tool to study nucleosome-level genome architectures 
## Installation
Method1: direct conda/mamba installation, need bioconda repo (mamba is recommended)
```
mamba create -n Nuc3DMap python=3
mamba install bwa samtools pairtools deeptools bedtools 
mamba install cooler pyranges psutil numba tqdm seaborn statsmodels pybedtools
mamba install anaconda::scikit-learn
pip install -U ray # if need to use parallel for NucLoad
cd /path/Nuc3DMap
pip install --editable .
```
Method2: use env.yaml through conda
```
conda env create -f <path to Nuc3DMap>/scripts/env.yml
cd /path/Nuc3DMap
pip install --editable .
```
Method3: use singularity (work in progress)

## Quick Start
**Note: Nuc3DMap would get poor performance when sequence depth is not enough under current version. We recommend at least 1b No-dup Pairs after Nuc3DMap nucprep (in foo.stats.summary.txt from nucprep)**  

**If input files contain fastq file, bwa index is needed before running Nuc3DMap nucprep**
```
wget https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz
gunzip GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz
bwa index GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta
```
**Recommendation, use trim_galore to trim the fastq file**
```
# Example code
# For paired-end 
trim_galore --trim-n --paired -j 5 -o ./ foo_R1.fastq.gz foo_R2.fastq.gz
# For single-end
trim_galore --trim-n -j 5 -o ./ foo.fastq.gz
```

#### Step1. Convert Hi-C and Micro-C fq/bam to .pairs and MNase bam to nuc_loc.bed file, do it as per-celltype wise, see the example file
```
Nuc3DMap nucprep -ifs <input_files.txt, see the example file in example_files/input_files.txt> \
                 -gs <hg38.chroms.size, hg38.sizes.genome in /data> \
                 -name <H1> \
                 -bfa </path/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta> \ #Run bwa index before using!
                 -inps <path/to/iNPS.py> \
                 -p <5>

e.g
Nuc3DMap nucprep -ifs input_h1.txt -gs hg38.chrom.sizes -name H1 -bfa GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta -inps ./Nuc3DMap/iNPS_V1.2.3.py -p 30

or run in background:
nohup Nuc3DMap nucprep -ifs input_h1.txt -gs hg38.chrom.sizes -name H1 -bfa GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta -inps ./Nuc3DMap/iNPS_V1.2.3.py -p 30 &> nucprep.log &
```
#### Step2. Build nucleosome-level contact map for Hi-C and Micro-C seperately, save as .cool file
```
Nuc3DMap nucload -pf <H1_HiC.pairs, from Nuc3DMap nucprep> \
                 -nucf <H1_nuc_loc.bed, from Nuc3DMap nucload> \
                 -od <./> \
                 -gt <hg38.annot.final.bed, hg38.annot.final.bed in /data> \
                 -refg <hg38> \
                 -np <5, ray installation needed, if use slurm or sbatch, use -np 1 for sake of compatibility>

e.g,
nohup Nuc3DMap nucload -pf H1_HiC_mapped.pairs -nucf H1_MNase_nucleosome_location.bed -od ./ -gt hg38.annot.final.bed -refg hg38 -np 20 &> h1.hic.log &
nohup Nuc3DMap nucload -pf H1_MicroC_mapped.pairs -nucf H1_MNase_nucleosome_location.bed -od ./ -gt hg38.annot.final.bed -refg hg38 -np 20 &> h1.mic.log &
```
#### Step3. Merge Hi-C and Micro-C nucleosome level contact map.
```
Nuc3DMap nucmerge -mc <./H1_MicroC.cool, from Nuc3DMap nucload> \
                  -hc <./H1_HiC.cool, from Nuc3DMap nucload> \
                  -edf <hg38.DpnII.bed, enzyme digestion file, ./data/hg38.DpnII.bed or ./data/hg38.HindIII.bed;
                        for other reference genomes or enzyme, please use Nuc3DMap_digest.py in utilites to generate>
                  -od <./>
                  -refg <hg38>
e.g.,
nohup Nuc3DMap nucmerge -mc ../nucload/H1_MicroC.1.0.cool -hc ../nucload/H1_HiC_mapped.1.0.cool -edf /data/kfang/NucOrg/Proj.110422/Nuc3DMap/data/hg38.HindIII.bed -od ./ -refg hg38 &> h1.nucmerge.log &
```
#### Step4. Call NucD(om)/NucB(oundary)/NucG(ap)
```
Nuc3DMap nuctd -imhc <./H1_iMHiC.cool>
                -ws <10 default 10>
                -od <./>
e.g.,
nohup Nuc3DMap nuctd -imhc ../nucmerge/H1_iMHiC_212025.cool -ws 10 -od ./ &> h1.nuctd.log &
```
#### Step5. Detect NucIL and NucL(oop)
```
Nuc3DMap nucil -imhc <./H1_iMHiC>.cool -t <./H1_iMHiC>.win10.txt -od ./ -np 20

e.g.,
nohup Nuc3DMap nucil -imhc ../nucmerge/H1_iMHiC_212025.cool -t ../nuctd/H1_iMHiC.win10.txt -od ./ -np 20 &>h1.nucil.log &
```
## Maunal
#### Nuc3DMap
```
Nuc3DMap --help

Usage: Nuc3DMap [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  nucprep   Convert .fq/bam into .pairs and nuc_loc.bed
  nucload   Build nucleosome-level contact map
  nucmerge  Merge Hi-C and Micro-C contact map
  nuctd     Calling NucDom/Boundary/Gap
  nucil     Detecting Nuc Interation Loci (NucIL)
```
#### Nuc3DMap nucprep
```
Usage: Nuc3DMap nucprep [OPTIONS]

  Convert .fq/bam into .pairs and nuc_loc.bed

Options:
  -ifs, --inputfiles PATH   Inputfiles, each row contains either fastq or bm
                            for a sample. Please refer to example_input check
                            details
  -gs, --gsize PATH         referencec genome size file, e.g, hg38.chrom.sizes
  -name, --prefixname TEXT  Specify the prefix name
  -mq, --minmapq INTEGER    minimum mapq for quality control
  -bfa, --bwafasta PATH     Required if input format is fastq, the path for
                            refernce fasta (bwa <idxbase>)
  -inps, --inpsdir PATH     Required if input has MNase-seq, the path of
                            iNPS.py. For example,
                            /data/NucHMM/scripts/iNPS_V1.2.2.py.
  -td, --tmpdir PATH        temporary folder to store intermediate results
  -p, --threads INTEGER     Number of threads, default 5
  --help                    Show this message and exit.
```
#### Nuc3DMap nucload
```
Usage: Nuc3DMap nucload [OPTIONS]

  Build nucleosome-level contact map

Options:
  -pf, --pairsfile PATH        Input pairs  [required]
  -nucf, --nuclocfile PATH     The nucleosome position files  [required]
  -gt, --geneannot PATH        reference genes annotation file  [required]
  -od, --outdir PATH           Specify the output files path and name.
  -refg, --refgenome TEXT      The name of reference genome, currently accept
                               hg19|hg38
  -rnolp, --reportnooverlap    Flag of saving pairs that are not directed
                               overlapped or overlapped length less than
                               cutoff
  -sr, --statsreport           Flag of plotting distributions of variables in
                               nucload
  -olpc, --overlapcut INTEGER  cutoff for overlapping length between pairs and
                               map bins, default 1 (auto mode is -1)
  -m, --match TEXT             best: one pair only belong to one bin | multi:
                               one pair belong can assgin to multiple bins
  -cs, --chunksize INTEGER     The chunksize to load the .pairs file
  -spl, --sampling FLOAT       Sampling pairs for robust test, ignore for
                               normal usage, [0,1]
  -np, --nproc INTEGER         The number of parallel processors
  --help                       Show this message and exit.
```
#### Nuc3DMap nucmerge
```
Usage: Nuc3DMap nucmerge [OPTIONS]

  Merge Hi-C and Micro-C contact map

Options:
  -mc, --microc PATH          MicroC.cool resulted from nucload.  [required]
  -hc, --hic PATH             HiC.cool resulted from nucload.  [required]
  -edf, --enzymedigestf PATH  enzyme digestion file  [required]
  -od, --outdir PATH          Specify the output files path and name.
  -refg, --refgenome TEXT     The name of reference genome, currently accept
                              hg19|hg38, default hg38
  -vis, --visualize           Visualize convergence and MD plot, which will
                              slow down the process
  -smf, --savetmpfile         Save temporary files
  -umf, --usetmpfile          Skipp first step, directly use temporary files
  --help                      Show this message and exit.
```
#### Nuc3DMap nuctd
```
Usage: Nuc3DMap nuctd [OPTIONS]

  Calling NucDom/Boundary/Gap

Options:
  -imhc, --imhiccool PATH  Integrated MicroC-HiC.cool resulted from nucmerge.
                           [required]
  -ws, --winsize INTEGER   window size for calling NucDom
  -od, --outdir PATH       Specify the output directory
  -ss, --shiftsize FLOAT   Empirical shift size for boundaries, default 0
  -rg, --rgmap             Input is renormalized grouping map
  --help                   Show this message and exit.
```
#### Nuc3DMap nucil
```
Usage: Nuc3DMap nucil [OPTIONS]

  Detecting Nuc Interation Loci (NucIL)

Options:
  -imhc, --imhiccool PATH       Integrated MicroC-HiC.cool resulted from
                                nucmerge.  [required]
  -t, --tads PATH               NucDom.txt from nucdom  [required]
  -rg, --rgmap PATH             If RGMap have built, use this command to skip
                                it
  -dup, --distanceup INTEGER    The maximum distance for detecting NucIL,
                                default 2Mb
  -dlow, --distancelow INTEGER  The maximum distance for detecting NucIL,
                                default 5kb
  -q, --fdr FLOAT               FDR cutoff for detecting NucIL, default 0.1
  -s, --sigma FLOAT             sigma for initial Gaussian smoothing , default
                                0.5
  -em, --extmode TEXT           Choose among None|constant|adjacent,
                                represents NoExt|Ext +/-3000|Ext adjacent
                                Nucdom.
  -od, --outdir PATH            Specify the output directory
  -np, --nproc INTEGER          The number of parallel processors
  --help                        Show this message and exit.
```
