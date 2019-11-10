# convert phased SNP file into PLINK formatted .map file
# SNP file is expected to follow format as downloaded from Hapmap3 project
# usage: ./toMAP.py -f [SNPfile] -c [chromosome number]
# output is directed to STDOUT by default

import argparse

parser = argparse.ArgumentParser(description='Convert SNP file from Hapmap3 project to PLINK formatted .map file')
parser.add_argument('-f', action="store", dest="f", type=str, help="path to SNP file")
parser.add_argument('-c', action="store", dest="c", type=int, help="chromosome number")

args = parser.parse_args()

with open(args.f, 'r') as SNP:
    line = SNP.readline()
    while line:
        rsID, pos, _ = line.strip().split('\t')
        print(f'{args.c}\t{rsID}\t{0}\t{pos}')


