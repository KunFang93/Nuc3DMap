import os, re
import subprocess
import pandas as pd
from Nuc3DMap_utilities import get_qc
def remove_extension_and_suffix(file_path):
    # Extract the file name from the path
    file_name = os.path.basename(file_path)
    
    # Handle multi-part extensions first
    if file_name.endswith('.fq.gz'):
        file_name = file_name[:-6]  # Remove 6 characters ('.fq.gz')
    elif file_name.endswith('.fastq.gz'):
        file_name = file_name[:-9]  # Remove 9 characters ('.fastq.gz')
    else:
        # For other extensions, use splitext
        base_name, ext = os.path.splitext(file_name)
        if ext in ['.fq', '.fastq', '.bam']:
            file_name = base_name

    # Remove _R1 suffix if present
    if file_name.endswith(('_R1','_r1','_R2','_r2')):
        file_name = file_name[:-3]  # Remove 3 characters ('_R1')
    return file_name

def prepfromfastq(fastq_r1, fastq_r2, prefix, ref_fasta, gsize, minmapq, tmpdir, nthreads):
    subprocess.call(f'bwa mem -5SP -T0 -t {nthreads} {ref_fasta} {fastq_r1} {fastq_r2} | '
                    f'pairtools parse2 --min-mapq {minmapq} --report-position read --report-orientation read --add-pair-index '
                    f'--add-columns pos5,pos3 --max-inter-align-gap 30 --nproc-in {nthreads} --nproc-out {nthreads} --chroms-path {gsize} | '
                    f'pairtools sort --tmpdir={tmpdir} --nproc {nthreads} | '
                    f'pairtools dedup --nproc-in {nthreads} --nproc-out {nthreads} --mark-dups --output-stats {prefix}.stats.txt | '
                    f'pairtools split --nproc-in {nthreads} --nproc-out {nthreads} --output-pairs {prefix}_mapped.pairs --output-sam - | '
                    f'samtools view -bS -@ {nthreads} | samtools sort -@ {nthreads} -o {prefix}_mapped.bam', shell=True)
    subprocess.call(f'samtools index {prefix}_mapped.bam', shell=True)
    get_qc(f'{prefix}.stats.txt', f'{prefix}.summary.txt')
    print("prepfromfastq Finished.")

def prepfrombam(bam_f, prefix, gsize, minmapq, tmpdir, nthreads):
    subprocess.call(f'pairtools parse2 --min-mapq {minmapq} --report-position read --report-orientation read --add-pair-index --add-columns pos5,pos3 '
                    f'--max-inter-align-gap 30 --nproc-in {nthreads} --nproc-out {nthreads} --chroms-path {gsize} {bam_f} | '
                    f'pairtools sort --tmpdir={tmpdir} --nproc {nthreads} | '
                    f'pairtools dedup --nproc-in {nthreads} --nproc-out {nthreads} --mark-dups --output-stats {prefix}.txt | '
                    f'pairtools split --nproc-in {nthreads} --nproc-out {nthreads} --output-pairs {prefix}_mapped.pairs -', shell=True)
    subprocess.call(f'samtools index {prefix}_mapped.bam', shell=True)
    get_qc(f'{prefix}.txt',f'{prefix}.summary.txt')
    print("prepfrombam Finished.")

def withinRegularChrom(ref_fasta, tmpdir):
    fai_df = pd.read_table(f'{ref_fasta}.fai',header=None)
    # remove random and scaffold
    fai_df = fai_df[~fai_df[0].str.contains('random|chrUn|chrEBV')]
    fai_df['start'] = 0
    fai_df = fai_df[[0,'start',1]]
    fai_df.to_csv(f'{tmpdir}/chroms.filt.bed',header=None,index=False,sep='\t')

def alignfq(line_fs, seqtype, prefix, minmapq, ref_fasta, tmpdir, nthreads):
    if seqtype == 'SE':  # single-end
        # Align reads
        subprocess.call(
            f'bwa mem -t {nthreads} {ref_fasta} {line_fs[0]} | '
            f'samtools view -@ {nthreads} -L {tmpdir}/chroms.filt.bed -F 1804 -q {minmapq} -bS | '
            f'samtools sort -@ {nthreads} -o {prefix}.bam',
            shell=True
        )
    elif seqtype == 'PE':  # paired-end
        # Align reads
        subprocess.call(
            f'bwa mem -t {nthreads} {ref_fasta} {line_fs[0]} {line_fs[1]} | '
            f'samtools view -@ {nthreads} -L {tmpdir}/chroms.filt.bed -F 1804 -f 2 -q {minmapq} -bS | '
            f'samtools sort -@ {nthreads} -o {prefix}.bam',
            shell=True
        )
    else:
        print("seqtype only limited to SE/PE")
        return None
    # Index the sorted BAM file
    subprocess.call(f'samtools index {prefix}.bam', shell=True)
    print("Alignfq Finished!")
    return seqtype

def nuc_calling(bamfile,pe,inpspath,tmpdir):
    # filter reads within the fragment size range [130,180]
    bamfile_body = re.findall('[^\/]+$',bamfile)[0][:-4]
    if pe=='PE':
        print("Screen out unfitted reads..")
        filt_bam = bamfile_body + '_screened.bam'
        if os.path.exists(bamfile+'.bai'):
            subprocess.call('alignmentSieve -p 4 -b ' + bamfile + ' -o ' + filt_bam +
                            ' --minFragmentLength 135 --maxFragmentLength 175',shell=True)
            subprocess.call(f'rm {bamfile}', shell=True)
        else:
            subprocess.call('samtools index -@ 4 '+ bamfile,shell=True)
            subprocess.call('alignmentSieve -p 4 -b ' + bamfile + ' -o ' + filt_bam +
                            ' --minFragmentLength 135 --maxFragmentLength 175',shell=True)
            subprocess.call(f'rm {bamfile}', shell=True)
    elif pe=='SE':
        filt_bam = bamfile
    else:
        print("Please indicate paired-end or single-end by PE or SE.")
        exit(1)
    # transfer bamtobed
    print("Transfer bam to bed..")
    bed_file = f'{tmpdirq}/{bamfile_body}.bed'
    if os.path.exists(bed_file):
        print("Use the existing bed file")
    else:
        subprocess.call('bedtools bamtobed -i ' + filt_bam + ' > ' + bed_file, shell=True)
    # use iNPS calling nucleosome core location
    print("iNPS Calling..")
    if pe=='PE':
        subprocess.call('python ' + inpspath + ' -i ' + bed_file +
                        ' -o nuc_calling_result/' + bamfile_body + ' --s_p p',shell=True)
    else:
        subprocess.call('python ' + inpspath + ' -i ' + bed_file +
                        ' -o nuc_calling_result/' + bamfile_body + ' --s_p s',shell=True)
    # remove header of Gathering.like_bed
    gathering_file = 'nuc_calling_result/' + bamfile_body + '_Gathering.like_bed'
    count = 1
    count2= 1
    with open(gathering_file,'r') as g_file:
        for line in g_file:
            line_info = line.strip().split()
            try:
                line_start = int(line_info[1])
            except ValueError:
                count += 1
                continue
            count2 += 1
            if count2 >100:
                break
    g_file.close()
    count += 1
    final_nuc_file = 'nuc_calling_result/' + bamfile_body + '_nucleosome_location.bed'
    subprocess.call('tail -n +' + str(count+1) + ' ' + gathering_file + ' > ' + final_nuc_file,shell=True)
    return final_nuc_file
