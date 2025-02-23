# $1 bam file, $2 out_prefix, $3 genome size file, $4 min_mapq, $5 number of processor
# $3: /data/kfang/Align_Index/grch38_no_alt/hg38.sizes.genome
pairtools parse2 --min-mapq $4 --report-position read --report-orientation read --add-pair-index --add-columns pos5,pos3 \
 --max-inter-align-gap 30 --nproc-in $5 --nproc-out $5 --chroms-path $3 $1 | \
pairtools sort --tmpdir=./ --nproc $5|pairtools dedup --nproc-in $5 \
--nproc-out $5 --mark-dups --output-stats "$2".txt |pairtools split --nproc-in $5 \
--nproc-out $5 --output-pairs "$2"_mapped.pairs -
