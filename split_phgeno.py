# split .phgeno file into one set for simulation (101 hyplotypes) and the rest for use as reference panel

import argparse
import sys
import numpy as np


def readEigenstrat(file):
    '''
    Read eigenstrat file. 
    Input:
        file: path to the eigenstrat file for a population. Each row represents a SNP, and each column represents a haplotype.
    Output:
        a numpy array version of the eigenstrat file.
        Similarly, each row is a SNP, and each column a haplotype.
    '''
    with open(file) as f:
        snps = f.readlines()
        snps = [list(snp) for snp in snps]
        return np.array(snps)[:,:-1]

def split(snpMatrix, numSimu):
    numSNP, numHap = snpMatrix.shape
    assert numSimu < numHap
    sel_col_simu = np.random.choice(numHap, numSimu, replace=False)
    np.savetxt('simu.phgeno',snpMatrix[:,sel_col_simu], delimiter='')
    np.savetxt('ref.phgeno', snpMatrix[:,np.setdiff1d(np.arange(numHap), sel_col_simu)], delimiter='')

def main():
    parser = argparse.ArgumentParser(description='split .phgeno file into one set for simulation (default 101 hyplotypes) and the rest for use as reference panel')
    parser.add_argument('-p', action="store", dest="p", type=str, 
                        help='path to .phgeno file.')
    parser.add_argument('-n', action="store", dest="n", type=int, default = 101, 
                        help='the number of haplotypes to use for simulating admixed individual')
    args = parser.parse_args()

    snpmatrix = readEigenstrat(args.p).astype(np.int32)
    split(snpmatrix, args.n)

    


if __name__ == '__main__':
    main()

