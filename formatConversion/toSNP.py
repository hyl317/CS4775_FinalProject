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
import copy

parser = argparse.ArgumentParser(description='prepare inputs to admi-simu program for two reference populations')
parser.add_argument('-p1', action="store", dest="p1", type=str, help="path to SNP file for pop1")
parser.add_argument('-p2', action="store", dest="p2", type=str, help="path to SNP file for pop2")
parser.add_argument('-m', action="store", dest='m', type=str, help="path to .map file as produced by insert_map.pl")
args = parser.parse_args()

with open(args.p1) as pop1, open(args.p2) as pop2, open(args.m) as map, \
    open('pop1.phgeno','w') as pop1out, open('pop2.phgeno','w') as pop2out, open('snp.txt','w') as snpout:
    # ignore header line in pop1 and pop2
    pop1.readline()
    pop2.readline()
    
    pop1_line, pop2_line, map_line = pop1.readline(), pop2.readline(), map.readline()
    
    while pop1_line and pop2_line and map_line:
        chrom, snpid, gen_dist, phy_loc = map_line.strip().split('\t')
        pop1_info, pop2_info = pop1_line.strip().split(' '), pop2_line.strip().split(' ')
        assert pop1_info[0] == pop2_info[0] and pop1_info[1] == pop2_info[1]

        pop1_snp, pop2_snp = copy.deepcopy(pop1_info[2:]), copy.deepcopy(pop2_info[2:])
        #print(pop1_snp)
        #print(pop2_snp)
        allele_set = list(set(pop1_snp + pop2_snp))

        # only biallelic position is retained
        if (len(allele_set)) != 2:
            pop1_line, pop2_line, map_line = pop1.readline(), pop2.readline(), map.readline()
            continue

        allele1, allele2 = allele_set[0], allele_set[1]
        #temp = map(lambda x: str(1) if x == allele1 else str(0), pop1_snp)
        #templist = list(map)
        #print(f'templist is {templist}')
        snpout.write(f'{snpid}\t{chrom}\t{gen_dist}\t{phy_loc}\t{allele1}\t{allele2}\n')
        #pop1out.write(''.join(map(lambda x: str(1) if x == allele1 else str(0), pop1_snp)))
        #pop1out.write(''.join(map(str, np.array(pop1_snp) == allele1)))
        pop1out.write(''.join(['1' if allele == allele1 else '0' for allele in pop1_snp]))
        pop1out.write('\n')
        #pop2out.write(''.join(map(lambda x: str(1) if x == allele1 else str(0), pop2_snp)))
        #pop2out.write(''.join(map(str, np.array(pop2_snp) == allele1)))
        pop1out.write(''.join(['1' if allele == allele1 else '0' for allele in pop2_snp]))
        pop2out.write('\n')

        pop1_line, pop2_line, map_line = pop1.readline(), pop2.readline(), map.readline()

