# Given .map file produced from insert_map, produce .snp file in the format required by admi-simu program.
# The .snp file format:
# [snp ID] [chromosome number] [genetic distance] [physical location in bps] [allele1] [allele2]
# In addition to .map, this program also takes in
# SNP file for two populations, in the format as provided in hapmap3 project
# Output:
# .phgeno file for the two populations. 1 indicates allel1, and 0 indicates allele2
# only biallelic SNPs are retained in the output.
# each row represents a SNP and each column a haplotype

import argparse
import numpy as np
import re
import os
import sys


parser = argparse.ArgumentParser(description='prepare inputs to admi-simu program for two reference populations. All files starting with hapmap3 in the directory will be extracted.')
parser.add_argument('-p1', action="store", dest="p1", type=str, help="path to the directory containing snp file for pop1")
parser.add_argument('-p2', action="store", dest="p2", type=str, help="path to the directory containing snp file for pop2")
parser.add_argument('-m', action="store", dest='m', type=str, help="path to .map file as produced by insert_map.pl")
args = parser.parse_args()

# preprocess SNP files for population
def extractSNParray(dir):
    # read all files starting with hapmap3 in the given directory
    # write the SNP into 2-d numpy array
    # each row is for one SNP and each column for one haplotype
    files = os.listdir(dir)
    snp_2dlist = []
    for file in files:
        if not file.startswith('hapmap3'):
            continue

        snp_array = [];
        with open(f'{dir}/{file}') as f:
            f.readline() #ignore header line
            snp_info = f.readline();
            while snp_info:
                snp_info = re.split('\s', snp_info.strip())
                snp_array.append(snp_info[2:])
                snp_info = f.readline()

        snp_2dlist.append(np.array(snp_array))
    return reduce(lambda x,y:np.concatenate(x,y,axis=1), snp_2dlist) # joining snp arrays from multiple hapmap3 files together

snparray1 = extractSNParray(args.p1)
snparray2 = extractSNParray(args.p2)

if snparray1.shape[0] != snparray2.shape[0]:
    print('Error: Population 1 and population 2 have different number of SNPs')
    sys.exit()


with open(args.m) as map, open('pop1.phgeno','w') as pop1out, \
    open('pop2.phgeno','w') as pop2out, open('snp.txt','w') as snpout:
    
    for i in range(snparray1.shape[0]):
        map_line = map.readline()
        if not map_line:
            print(f'genetic map information is lacking for snp {i+1}')
            sys.exit()

        chrom, snpid, gen_dist, phy_loc = map_line.strip().split('\t')
        snp_pop1, snp_pop2 = snparray1[i], snparray2[i]
        snpset = set(snp_pop1+snp_pop2)

        if len(snpset) != 2: # only biallelic SNP is retained for downstream analysis
            continue

        allele1, allele2 = snpset[0], snpset[2]
        snpout.write(f'{snpid}\t{chrom}\t{gen_dist}\t{phy_loc}\t{allele1}\t{allele2}\n')
        pop1out.write(''.join(['1' if allele == allele1 else '0' for allele in snp_pop1]))
        pop1out.write('\n')
        pop2out.write(''.join(['1' if allele == allele1 else '0' for allele in snp_pop2]))
        pop2out.write('\n')

