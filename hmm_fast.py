import numpy as np
import math
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

    def transition(self, r):
        noAncestrySwitch = math.exp(-r*self.t)
        noRecomb1, noRecomb2 = math.exp(-r*self.rho1), math.exp(-r*self.rho2)
        return noAncestrySwitch, noRecomb1, noRecomb2

    def emission(self, obs_j, j):
        # return log probability of emission for site j
        pop1SNP, pop2SNP = self.pop1matrix[j][:,np.newaxis], self.pop2matrix[j][:,np.newaxis]
        pop1 = np.concatenate((pop1SNP == obs_j, pop1SNP != obs_j), axis=1)
        pop2 = np.concatenate((pop2SNP == obs_j, pop2SNP != obs_j), axis=1)
        theta_pop1 = np.array([1-self.theta1, self.theta1])[:,np.newaxis]
        theta_pop2 = np.array([1-self.theta2, self.theta2])[:,np.newaxis]
        emission = np.concatenate((pop1@theta_pop1, pop2@theta_pop2))
        return np.log(emission)

    def emissionALL(self, obs):
        # precompute all emission probabilities for all sites
        # each SNP occupies a row, and each column correspond to a state
        emis = np.zeros((self.numSNP, self.n1+self.n2))
        for j in range(self.numSNP):
            emis[j] = self.emission(obs[j], j).flatten()
        return emis

        
    @jit
    def forward(self, emis, nrow, ncol):
        # Given the observed haplotype, compute its forward matrix
        f = np.zeros((nrow, ncol))
        # initialization
        f[:,0] = (self.initial + emis[0]).flatten()
        
         # fill in forward matrix
        for j in range(1, ncol):
            noAncestrySwitch, noRecomb1, noRecomb2 = self.transition(self.D[j])
            common_pop1 = logsumexp(np.log(1-noAncestrySwitch) + np.log(self.mu) - np.log(self.n1) + f[:,j-1])
            common_pop2 = logsumexp(np.log(1-noAncestrySwitch) + np.log(1-self.mu) - np.log(self.n2) + f[:,j-1])
            #next term is for cases where i=l
            term1_pop1 = logsumexp(np.log(noAncestrySwitch) + np.log(1-noRecomb1) - np.log(self.n1) + f[:self.n1, j-1])
            term1_pop2 = logsumexp(np.log(noAncestrySwitch) + np.log(1-noRecomb2) - np.log(self.n2) + f[self.n1:, j-1])
            #last term is only for i=l and n=k
            term2_pop1 = np.log(noAncestrySwitch) + np.log(noRecomb1) + f[:self.n1, j-1]
            term2_pop2 = np.log(noAncestrySwitch) + np.log(noRecomb2) + f[self.n1:, j-1]

            temp = np.concatenate((np.repeat(common_pop1, self.n1), np.repeat(common_pop2, self.n2)), axis=0)
            #print(temp.shape)
            temp[:self.n1] = np.apply_along_axis(np.logaddexp, 0, np.repeat(term1_pop1, self.n1), temp[:self.n1])
            temp[self.n1:] = np.apply_along_axis(np.logaddexp, 0, np.repeat(term1_pop2, self.n2), temp[self.n1:])
            #print(temp.shape)
            temp[:self.n1] = np.apply_along_axis(np.logaddexp, 0, term2_pop1, temp[:self.n1])
            temp[self.n1:] = np.apply_along_axis(np.logaddexp, 0, term2_pop2, temp[self.n1:])

            # using axis=0, logsumexp sum over each column of the transition matrix
            f[:, j] = emis[j] + temp
        return f


    #@jit
    def backward(self, emis, nrow, ncol):
        # Given the observed haplotype, compute its backward matrix
        b = np.zeros((nrow, ncol))
        # initialization
        b[:, ncol-1] = np.full(nrow, 0)

        for j in range(ncol-2, -1, -1):
            T = self.transition(self.D[j+1])
            b[:,j] = logsumexp(T + emis[j+1] + b[:,j+1], axis=1)
        return b
    
    #@jit
    def posterior(self, f, b, n1, n2, ncol):
        # posterior decoding
        post = np.zeros((n1+n2, ncol))
        for j in range(ncol):
            log_px = logsumexp(f[:,j] + b[:,j])
            post[:,j] = np.exp(f[:,j] + b[:,j] - log_px) 

        post_pop1, post_pop2 = post[:n1], post[n1:n1 + n2]
        post_pop1, post_pop2 = np.sum(post_pop1, axis=0), np.sum(post_pop2, axis=0)
        return post_pop1, post_pop2
    

    def decode(self, obs):
        # infer hidden state of each SNP sites in the given haplotype
        # state[j] = 0 means site j was most likely copied from population 1 
        # and state[j] = 1 means site j was most likely copies from population 2

        start = time.time()
        emis = self.emissionALL(obs)
        n1, n2 = self.n1, self.n2
        ncol = self.numSNP
        f = self.forward(emis, n1+n2, ncol)
        #b = self.backward(emis, n1+n2, ncol)
        end = time.time()
        print(f'uncached version takes time {end-start}')
        print(f'forward probability:{logsumexp(f[:,-1])}')
        #print(f'backward probability:{logsumexp(self.initial + emis[0] + b[:,0])}')

        #post_pop1, post_pop2 = self.posterior(f,b, n1, n2, ncol)
        #return [0 if prob1 > prob2 else 1 for prob1, prob2 in zip(post_pop1, post_pop2)]



