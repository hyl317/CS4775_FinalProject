import numpy as np
import math
import multiprocessing
from numba import jit
from scipy.special import logsumexp
import time

class HMM(object):
    def __init__(self, pop1_snp, pop2_snp, mu, t, numSNP, n1, n2, rho1, rho2, theta1, theta2, D):
        self.pop1matrix = pop1_snp
        self.pop2matrix = pop2_snp
        self.mu = mu
        self.t = t
        self.numSNP = numSNP
        self.n1, self.n2 = n1, n2
        self.rho1, self.rho2 = rho1, rho2
        self.theta1, self.theta2 = theta1, theta2
        self.D = D
        self.initial = np.log(np.array([mu/n1]*n1 + [(1-mu)/n2]*n2))

    #@profile
    def transition(self, r):
        # calculate transition log probability matrix between two sites separated by distance r (measured in Morgan)
        numHiddenState = self.n1 + self.n2
        T = np.full((numHiddenState, numHiddenState), np.nan)
        # transition: (i, k) -> (l, n)
        # i,l refers to which ancestral population this SNP is drawn
        # k,n refers to which haplotype from population i/l this SNP is drawn
        # we use 0 to represent population 1 and 1 to represent population 2

        # For cases where i=l and k=n (aka diagonal entries)
        # case1: i=l=0
        term01 = -r*self.t - self.rho1*r
        term02 = math.log(1-math.exp(-r*self.t)) + math.log(self.mu) - math.log(self.n1)
        term03 = -r*self.t + math.log(1-math.exp(-self.rho1*r)) - math.log(self.n1)
        diag0 = logsumexp([term01, term02, term03])

        # case2: i=l=1
        term11 = -r*self.t - self.rho2*r
        term12 = math.log(1-math.exp(-r*self.t)) + math.log(1-self.mu) - math.log(self.n2)
        term13 = -r*self.t + math.log(1-math.exp(-self.rho2*r)) - math.log(self.n2)
        diag1 = logsumexp([term11, term12, term13])


        # For cases where there is a 'successfully' (aka, i != l) ancestry switch
        # swtich to ancestry 0
        AncestrySwitchTo0 = math.log(1-math.exp(-r*self.t)) +  math.log(self.mu) - math.log(self.n1)
        # switch to ancestry 1
        AncestrySwitchTo1 = math.log(1-math.exp(-r*self.t)) + math.log(1-self.mu) - math.log(self.n2)

        # for cases where there is no ancestry switch (or a silent ancestry switch)
        # but a 'successful' (aka i=l && k != n) haplotype switch within the same ancestry
        # case1: i=l=0
        term01 = math.log(1-math.exp(-r*self.t)) + math.log(self.mu) - math.log(self.n1)
        term02 = -r*self.t + math.log(1-math.exp(-self.rho1*r)) - math.log(self.n1)
        haploSwitchAncestry0 = logsumexp([term01, term02])
        # case2: i=l=1
        term11 = math.log(1-math.exp(-r*self.t)) + math.log(1-self.mu) - math.log(self.n2)
        term12 = -r*self.t + math.log(1-math.exp(-self.rho2*r)) - math.log(self.n2)
        haploSwitchAncestry1 = logsumexp([term11, term12])

        T[:self.n1, :self.n1] = haploSwitchAncestry0
        T[self.n1:, self.n1:] = haploSwitchAncestry1
        T[:self.n1, self.n1:] = AncestrySwitchTo1
        T[self.n1:, :self.n1] = AncestrySwitchTo0
        # lastly, fill in the diagonal entries
        diagonal = [diag0]*self.n1 + [diag1]*self.n2
        di = np.diag_indices(self.n1 + self.n2)
        T[di] = diagonal
        
        return T

    #@profile
    #@jit(nopython=True)
    def emission(self, obs_j, j):
        # return log probability of emission for site j
        pop1SNP, pop2SNP = self.pop1matrix[j][:,np.newaxis], self.pop2matrix[j][:,np.newaxis]
        pop1 = np.concatenate((pop1SNP == obs_j, pop1SNP != obs_j), axis=1)
        pop2 = np.concatenate((pop2SNP == obs_j, pop2SNP != obs_j), axis=1)
        theta_pop1 = np.array([1-self.theta1, self.theta1])[:,np.newaxis]
        theta_pop2 = np.array([1-self.theta2, self.theta2])[:,np.newaxis]
        emission = np.concatenate((pop1@theta_pop1, pop2@theta_pop2))
        return np.log(emission)

    #@profile
    #@jit(nopython=True)
    def emissionALL(self, obs):
        # precompute all emission probabilities for all sites
        # each SNP occupies a row, and each column correspond to a state
        emis = np.zeros((self.numSNP, self.n1+self.n2))
        for j in range(self.numSNP):
            emis[j] = self.emission(obs[j], j).flatten()
        return emis

        
    #@profile
    @jit
    def forward(self, emis, nrow, ncol):
        # Given the observed haplotype, compute its forward matrix
        f = np.zeros((nrow, ncol))
        # initialization
        f[:,0] = (self.initial + emis[0]).flatten()
        
         # fill in forward matrix
        for j in range(1, ncol):
            T = self.transition(self.D[j])
            # using axis=0, logsumexp sum over each column of the transition matrix
            f[:, j] = emis[j] + logsumexp(f[:,j-1][:,np.newaxis] + T, axis=0)
        return f


    #@profile
    @jit
    def backward(self, emis, nrow, ncol):
        # Given the observed haplotype, compute its backward matrix
        b = np.zeros((nrow, ncol))
        # initialization
        b[:, ncol-1] = np.full(nrow, 0)

        for j in range(ncol-2, -1, -1):
            T = self.transition(self.D[j+1])
            b[:,j] = logsumexp(T + emis[j+1] + b[:,j+1], axis=1)
        return b
    
    @jit
    def posterior(self, f, b, n1, n2, ncol):
        # posterior decoding
        post = np.zeros((n1+n2, ncol))
        for j in range(ncol):
            log_px = logsumexp(f[:,j] + b[:,j])
            post[:,j] = np.exp(f[:,j] + b[:,j] - log_px) 

        post_pop1, post_pop2 = post[:n1], post[n1:n1 + n2]
        post_pop1, post_pop2 = np.sum(post_pop1, axis=0), np.sum(post_pop2, axis=0)
        return post_pop1, post_pop2
    
    #@profile
    def decode(self, obs):
        # infer hidden state of each SNP sites in the given haplotype
        # state[j] = 0 means site j was most likely copied from population 1 
        # and state[j] = 1 means site j was most likely copies from population 2

        start = time.time()
        emis = self.emissionALL(obs)
        n1, n2 = self.n1, self.n2
        ncol = self.numSNP
        f = self.forward(emis, n1+n2, ncol)
        b = self.backward(emis, n1+n2, ncol)
        end = time.time()
        print(f'uncached version takes time {end-start}')
        print(f'forward probability:{logsumexp(f[:,-1])}')
        print(f'backward probability:{logsumexp(self.initial + emis[0] + b[:,0])}')

        post_pop1, post_pop2 = self.posterior(f,b, n1, n2, ncol)
        return [0 if prob1 > prob2 else 1 for prob1, prob2 in zip(post_pop1, post_pop2)]

