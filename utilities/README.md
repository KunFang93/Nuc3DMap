# Useful scripts for preparing input files for Nuc3DMap
## For NucLoad
Ideally, NucPrep could handle all prep processes from scratch for HiC, Micro-C and MNase (e.g, from fastq). But in case cases, users might already have bam files or only want to process one types of the data. So the scripts contains in this folder leverage flexibility for users. 

Make sure running all following script under Nuc3DMap environment

MHiC_bam_process.sh can generate HiC or MicroC pairs from bam file, usage:
```
sh MHiC_bam2pairs.sh foo.bam foo hg38.genome.size 30 16 # input.bam prefix genome.size min_mapq nproc
```

## For NucMerge
Digest the reference genome by the provided restriction enzymes(s) and generate a BED file with the list of restriction fragments after digestion.
The output file can then be used by Nuc3DMap nucmerge -edf.
Note that the cutting site of the restriction enzyme has to be specified using the ‘^’ character.
The restriction enzymes HindIII, DpnII and BglII are encoded within the script and are therefore recognized if specified to the program.
Finally, multiple restriction enzymes can also be provided.
```
## Make sure the fasta file is the same one used for Nuc3DMap nucprep
## Digest the hg38 genome by HindIII
python Nuc3DMap_digest.py -r A^AGCTT -o hg38_hindiii.bed hg38.fasta

## The same ...
python Nuc3DMap_digest.py -r hindiii -o hg38_hindiii.bed hg38.fasta

## Double digestion, HindIII + DpnII
python Nuc3DMap_digest.py -r hindiii dpnii -o hg38_hindiii_dpnii.bed hg38.fasta
```
