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

def main():
    parser = argparse.ArgumentParser(description='Local Ancestry Inference as implemented in Hapmix.')
    parser.add_argument('-p1', action="store", dest="p1", type=str, help='eigenstrat SNP file for ancestral population 1 (eg. CEU)')
    parser.add_argument('-p2', action="store", dest="p2", type=str, help='eigenstrat SNP file for ancestral population 2 (eg. YRI)')
    parser.add_argument('-a', action="store", dest="a", type=str, help='eigenstrat SNP file for the admixed population.')
    parser.add_argument('-mu', action="store", dest='mu', type=float, default=0.2, help='percentage of genetic composition of population 1 in the admixed population')
    parser.add_argument('-t', action="store", dest='t', type=int, default=6, help='Time (in numbers of generation) to the admixture event')
    args = parser.parse_args()

    pop1_snp, pop2_snp, a_snp = readEigenstrat(args.p1), readEigenstrat(args.p2), readEigenstrat(args.a)

    # check whether the input data contains the same number of SNP
    if not (pop1_snp.shape[0] == pop2_snp.shape[0] and pop2_snp.shape[0] == a_snp.shape[0]):
        print('Input data should contain the same set of SNPs. \nExitting......')
        sys.exit()

    # set parameters of HMM as suggested in the original paper
    numSNP = pop1_snp.shape[0]
    n1, n2 = pop1_snp.shape[1], pop2_snp.shape[1]
    rho1, rho2 = 60000/n1, 90000/n2
    theta1, theta2 = 0.2/(0.2+n1), 0.2/(0.2+n2)
    print('data preprocessing done. Ready to run HMM model')
    print(f'Input data contains {} SNP sites')
    print(f'mu1={args.mu},T={args.T}')
    print(f'rho1={rho1},rho2={rho2}')
    print(f'theta1={theta1},theta2={theta2}')


if __name__ == '__main__':
    main()