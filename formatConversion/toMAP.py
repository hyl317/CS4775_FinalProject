# convert phased SNP file into PLINK formatted .map file
# SNP file is expected to follow format as downloaded from Hapmap3 project
# usage: ./toMAP.py -f [SNPfile] -c [chromosome number]
# output is directed to STDOUT by default
# In the output, the third column is supposed to be genetic distance
# which should be calibrated with genetic map downloaded from Hapmap3 project FTP site
# if not, this output provides a rough estimate for recombination rate

import argparse

parser = argparse.ArgumentParser(description='Convert SNP file from Hapmap3 project to PLINK formatted .map file')
parser.add_argument('-f', action="store", dest="f", type=str, help="path to SNP file")
parser.add_argument('-c', action="store", dest="c", type=int, help="chromosome number")

args = parser.parse_args()

with open(args.f, 'r') as SNP:
    line = SNP.readline() #ignore the header line
    line = SNP.readline()
    while line:
        #print(line.strip())
        #print(line.strip().split(' '))
        rsID, pos, *_ = line.strip().split(' ')
        print(f'{args.c}\t{rsID}\t{int(pos)/100000000}\t{pos}')
        line = SNP.readline()


