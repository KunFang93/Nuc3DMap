# Example data
Small T47D dataset to test the **nucprep** (since iNPS will go through all chroms, the run time of the example data will cost ~3h)
1. Download example data
```
wget https://www.dropbox.com/scl/fi/auyo6t3mmr9bv4468zmfh/T47D_MicroC_rep1_R1.sample10M.fastq.gz?rlkey=ak8ftrs46wowwfbpoka7asvff&st=ynnouieo -O T47D_MicroC_rep1_R1.sample10M.fastq.gz
wget https://www.dropbox.com/scl/fi/1oootdtgke487mv6gn6pj/T47D_MicroC_rep1_R2.sample10M.fastq.gz?rlkey=jlfnfbsfpxe6ve66i9bod8ebc&st=f4ccdrvw -O T47D_MicroC_rep1_R2.sample10M.fastq.gz
wget https://www.dropbox.com/scl/fi/oubj2l8m3l0e5j0gjdf0g/T47DTR_Mnase_rep1.sample10M.fastq.gz?rlkey=6imzkaooxavhasfo0065lx5nh&st=4c4ufklb -O T47DTR_Mnase_rep1.sample10M.fastq.gz
```
2. Use input_list.t47d.txt and Nucprep command
```
Nuc3DMap nucprep -ifs input_list.txt -gs <path>/hg38.chrom.sizes -bfa <path>/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta -inps <path>/iNPS_V1.2.3.py -p 10
```
**Note: this example file only show the input format for nucprep, it won't generate meaningful results for other modules**
