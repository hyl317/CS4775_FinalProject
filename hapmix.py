import argparse
import sys
import numpy as np
import hmm
import hmm_fast
import hmm_miscopy
from numba import jit

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

def readGeneticMap(file):
    '''
    Read genetic map file.
    The input file should follow the following format:
    [snpID]\t[chromosome number]\t[genetic distance]\t[physical distance]\t[allele1]\t[allele2]
    Output:
        A list D containing genetic distance between consecutive SNP sites.
        For example, D[j] should be the genetic distance between the jth and (j+1)th SNPs.
        zero-indexed.

        A list p containing physical location of each SNP site.
    '''
    with open(file) as f:
        d = [0]
        d.extend([line.strip().split('\t')[2] for line in f.readlines()])
        d = list(map(float, d))
        D = [d[i+1]-d[i] for i in range(len(d)-1)]

        p = [line.strip().split('\t')[3] for line in f.readlines()]
        p = list(map(int, p))
        return np.array(D), np.array(p)

def main():
    parser = argparse.ArgumentParser(description='Local Ancestry Inference as implemented in Hapmix.')
    parser.add_argument('-p1', action="store", dest="p1", type=str, help='eigenstrat SNP file for ancestral population 1 (eg. CEU)')
    parser.add_argument('-p2', action="store", dest="p2", type=str, help='eigenstrat SNP file for ancestral population 2 (eg. YRI)')
    parser.add_argument('-a', action="store", dest="a", type=str, help='eigenstrat SNP file for the admixed population.')
    parser.add_argument('-m', action="store", dest='m', type=str, help='genetic map. i.e, the .snp file used in admixture simulation.')
    parser.add_argument('-mu', action="store", dest='mu', type=float, default=0.2, help='percentage of genetic composition of population 1 in the admixed population')
    parser.add_argument('-t', action="store", dest='t', type=int, default=6, help='Time (in numbers of generation) to the admixture event')
    parser.add_argument('--miscopy', action="store", dest='miscopy', type=float, default=0.05, help="miscopying probability. Default is 0.05.")
    parser.add_argument('--mis', action="store_true", dest='mis', help="If this flag is asserted, then miscopy will be allowed.")
    args = parser.parse_args()

    pop1_snp, pop2_snp, a_snp = readEigenstrat(args.p1), readEigenstrat(args.p2), readEigenstrat(args.a)
    D, P = readGeneticMap(args.m)
    # check whether the input data contains the same number of SNP
    if not (pop1_snp.shape[0] == pop2_snp.shape[0] and pop2_snp.shape[0] == a_snp.shape[0] \
        and a_snp.shape[0] == len(D)):
        print('Input data should contain the same set of SNPs. \nExitting...')
        sys.exit()

    # check validity of genetic distance
    if np.any(D <= 0):
        print('Genetic distance between two consecutive site should be greater than zero.\n Exitting...')
        sys.exit()

    # set parameters of HMM as suggested in the original paper
    numSNP = pop1_snp.shape[0]
    n1, n2 = pop1_snp.shape[1], pop2_snp.shape[1]
    rho1, rho2 = 60000/n1, 90000/n2
    theta1, theta2 = 0.2/(0.2+n1), 0.2/(0.2+n2)
    print('data preprocessing done. Ready to run HMM model')
    print(f'Input data contains {numSNP} SNP sites')
    print(f'mu1={args.mu},T={args.t}')
    print(f'rho1={rho1},rho2={rho2}')
    print(f'theta1={theta1},theta2={theta2}')

    hmmModel = hmm_fast.HMM(pop1_snp, pop2_snp, args.mu, args.t, numSNP, n1, n2, rho1, rho2, theta1, theta2, D)
    if args.mis:
        hmmModel = hmm_miscopy.HMM_mis(pop1_snp, pop2_snp, args.mu, args.t, numSNP, n1, n2, rho1, rho2, theta1, theta2, D, args.miscopy)
    
    posterior = np.zeros((a_snp.shape[1], numSNP)) #row is for each haplotype, column is for each SNP site
    with open('decode.numba.fast.txt','w') as output:
        # the raw file prints posterior probability for each snp site to be in population1
        # one haplotype for line, one column per SNP site
        for i in range(a_snp.shape[1]):
            post1, post2 = hmmModel.decode(a_snp[:, i])
            break
            #posterior[i] = post1
            #states = [0 if prob1 > prob2 else 1 for prob1, prob2 in zip(post1, post2)]
            # find ancestry switching point
            #prev  = [states[0]] + states[:numSNP-1]
            #diff = np.array(states) - np.array(prev)
            #switch_points = np.where(diff != 0)[0]
            #print(switch_points)
            #report = ''
            #for point in switch_points:
            #    report += f'{prev[point]}:{point-1} '
        
            #if np.all(diff == 0) or switch_points[-1] != numSNP-1:
            #    report += f'{states[-1]}:{numSNP-1}'
            #output.write(f'{report}\n')

    #np.savetxt('raw.posterior.gz', posterior, delimiter='\t')


if __name__ == '__main__':
    main()
