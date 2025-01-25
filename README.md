# Nuc3DMap
A comprehensive tool to study nucleosome-level genome architectures 
## Installation
Method1: direct conda installation, need bioconda repo
```
conda create -n Nuc3DMap python=3
conda install bwa samtools pairtools deeptools bedtools
conda install cooler pyranges psutil numba tqdm seaborn statsmodels
conda install anaconda::scikit-learn
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
**If input files contain fastq file, bwa index is needed before running Nuc3DMap nucprep**
```
wget https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz
gunzip GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz
bwa index GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta
```
### Step1. Convert Hi-C and Micro-C fq/bam to .pairs and MNase bam to nuc_loc.bed file, do it as per-celltype wise, see the example file
```
Nuc3DMap nucprep -ifs <input_files.txt, see the example file in example_files/input_files.txt> \
                 -gs <hg38.chroms.size, hg38.sizes.genome in /data> \
                 -name <H1> \
                 -bfa </path/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta> \ #Run bwa index before using!
                 -inps <path/to/iNPS.py> \
                 -p <5>

e.g
Nuc3DMap nucprep -ifs input_mcf7.txt -gs hg38.chrom.sizes -name MCF7 -bfa GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta -inps ./Nuc3DMap/iNPS_V1.2.3.py -p 30

or run in background:
nohup Nuc3DMap nucprep -ifs input_mcf7.txt -gs hg38.chrom.sizes -name MCF7 -bfa GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta -inps ./Nuc3DMap/iNPS_V1.2.3.py -p 30 &> nucprep.log &
```
### Step2. Build nucleosome-level contact map for Hi-C and Micro-C seperately, save as .cool file
```
Nuc3DMap nucload -pf <H1_HiC.pairs, from Nuc3DMap nucprep> \
                 -nucf <H1_nuc_loc.bed, from Nuc3DMap nucload> \
                 -od <./> \
                 -gt <hg38.annot.final.bed, hg38.annot.final.bed in /data> \
                 -refg <hg38> \
                 -np <5, ray installation needed, if use slurm or sbatch, use -np 1 for sake of compatibility>

e.g,
nohup Nuc3DMap nucload -pf MCF7_HiC_mapped.pairs -nucf nucprep/nuc_calling_result/MCF7_MNase_nucleosome_location.bed -od ./ -gt hg38.annot.final.bed -refg hg38 -np 20 &> mcf7.hic.log &
nohup Nuc3DMap nucload -pf MCF7_MicroC_mapped.pairs -nucf nucprep/nuc_calling_result/MCF7_MNase_nucleosome_location.bed -od ./ -gt hg38.annot.final.bed -refg hg38 -np 20 &> mcf7.mic.log &
```
#### Step3. Merge Hi-C and Micro-C nucleosome level contact map.
```
Nuc3DMap nucmerge -mc <./H1_MicroC.cool, from Nuc3DMap nucload> \
                  -hc <./H1_HiC.cool, from Nuc3DMap nucload> \
                  -edf <hg38.DpnII.bed, enzyme digestion file, ./data/hg38.DpnII.bed or ./data/hg38.HindIII.bed;
                        for other reference genomes or enzyme, please use Nuc3DMap_digest.py in utilites to generate>
                  -od <./>
                  -refg <hg38>
```
### Step4. Call NucDom
```
Nuc3DMap nucdom -imhc <./H1_iMHiC.cool>
                -ws <10 default 10>
                -od <./>
```

## Maunal
Work in Progress
