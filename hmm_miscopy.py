import numpy as np
import math
from numba import jit
from scipy.special import logsumexp
import time

class HMM_mis(object):
    def __init__(self, pop1_snp, pop2_snp, mu, t, numSNP, n1, n2, rho1, rho2, theta1, theta2, D, miscopy):
        self.pop1matrix = pop1_snp
        self.pop2matrix = pop2_snp
        self.mu = mu
        self.t = t
        self.numSNP = numSNP
        self.n1, self.n2 = n1, n2
        self.rho1, self.rho2 = rho1, rho2
        self.theta1, self.theta2, self.theta3 = theta1, theta2, 0.01
        self.D = D
        self.miscopy = miscopy
        initialprob = [mu*(1-self.miscopy)/n1]*n1 + [mu*self.miscopy/n2]*n2 + [(1-mu)*self.miscopy/n1]*n1 + [(1-mu)*(1-self.miscopy)/n2]*n2
        self.initial = np.log(np.array(initialprob))

    def transition(self, r):
        noAncestrySwitch = math.exp(-r*self.t)
        noRecomb1, noRecomb2 = math.exp(-r*self.rho1), math.exp(-r*self.rho2)
        return noAncestrySwitch, noRecomb1, noRecomb2

    def emission(self, obs_j, j):
        # return log probability of emission for site j
        # first let's compute emission for non-miscopy case (i=j in the triplet)
        pop1SNP, pop2SNP = self.pop1matrix[j][:,np.newaxis], self.pop2matrix[j][:,np.newaxis]
        indicator_pop1 = np.concatenate((obs_j == pop1SNP, obs_j != pop1SNP), axis=1)
        indicator_pop2 = np.concatenate((obs_j == pop2SNP, obs_j != pop2SNP), axis=1)
        theta1 = np.array([1-self.theta1, self.theta1])[:,np.newaxis]
        theta2 = np.array([1-self.theta2, self.theta2])[:,np.newaxis]
        theta3 = np.array([1-self.theta3, self.theta3])[:, np.newaxis]
        emission = np.concatenate((indicator_pop1@theta1, indicator_pop2@theta3, 
                                   indicator_pop1@theta3, indicator_pop2@theta2))
        return np.log(emission)


    def emissionALL(self, obs):
        # precompute all emission probabilities for all sites
        # each SNP occupies a row, and each column correspond to a state
        emis = np.zeros((self.numSNP, 2*(self.n1+self.n2)))
        for j in range(self.numSNP):
            emis[j] = self.emission(obs[j], j).flatten()
        return emis

        
    @jit(parallel=True)
    def forward(self, emis):
        logMiscopy, logNoMiscopy = np.log(self.miscopy), np.log(1-self.miscopy)
        f = np.zeros((2*(self.n1+self.n2), self.numSNP))
        # initialization
        f[:,0] = self.initial + emis[0]
        #print(f"initial:{self.initial}")
        #print(f"f[:,0]:{f[:,0]}")
        for j in range(1, self.numSNP):
            noAncestrySwitch, noRecomb1, noRecomb2 = self.transition(self.D[j])
            
            # first let's deal with m=l case
            term1_pop1 = logsumexp(np.log(1-noAncestrySwitch) + np.log(self.mu) + logNoMiscopy - 
                                   np.log(self.n1) + f[:,j-1])
            term1_pop2 = logsumexp(np.log(1-noAncestrySwitch) + np.log(1-self.mu) + logNoMiscopy - 
                                   np.log(self.n2) + f[:, j-1])
            term2_pop1 = logsumexp(np.log(noAncestrySwitch) + np.log(1-noRecomb1) + logNoMiscopy - 
                                   np.log(self.n1) + f[:self.n1+self.n2, j-1])
            term2_pop2 = logsumexp(np.log(noAncestrySwitch) + np.log(1-noRecomb2) + logNoMiscopy - 
                                   np.log(self.n2) + f[self.n1+self.n2:,j-1])
            term3_pop1 = np.log(noAncestrySwitch) + np.log(noRecomb1) + f[:self.n1, j-1]
            term3_pop2 = np.log(noAncestrySwitch) + np.log(noRecomb2) + f[2*self.n1+self.n2:, j-1]
            #print(f"{term1_pop1}\n{term1_pop2}\n{term2_pop1}\n{term2_pop2}\n{term3_pop1}\n{term3_pop2}")
            # term1
            f[:self.n1, j] = np.repeat(term1_pop1, self.n1)
            f[2*self.n1+self.n2:, j] = np.repeat(term1_pop2, self.n2)
            # term2
            f[:self.n1, j] = np.apply_along_axis(np.logaddexp, 0, np.repeat(term2_pop1, self.n1), f[:self.n1, j])
            f[2*self.n1+self.n2:, j] = np.apply_along_axis(np.logaddexp, 0, np.repeat(term2_pop2, self.n2), f[2*self.n1+self.n2:, j])
            # term3
            f[:self.n1, j] = np.apply_along_axis(np.logaddexp, 0, term3_pop1, f[:self.n1, j])
            f[2*self.n1+self.n2:, j] = np.apply_along_axis(np.logaddexp, 0, term3_pop2, f[2*self.n1+self.n2:, j])

            # now let's deal with m != l case
            # here the pop1 and pop2 refers to the second entry in the triplet 
            term1_pop1 = logsumexp(np.log(1-noAncestrySwitch) + logMiscopy + np.log(1-self.mu) - np.log(self.n1) + f[:,j-1])
            term1_pop2 = logsumexp(np.log(1-noAncestrySwitch) + logMiscopy + np.log(self.mu) - np.log(self.n2) + f[:,j-1])
            term2_pop1 = logsumexp(np.log(noAncestrySwitch) + np.log(1-noRecomb2) + logMiscopy - 
                                   np.log(self.n1) + f[self.n1+self.n2:, j-1])
            term2_pop2 = logsumexp(np.log(noAncestrySwitch) + np.log(1-noRecomb1) + logMiscopy -
                                   np.log(self.n2) + f[:self.n1+self.n2, j-1])
            term3_pop1 = np.log(noAncestrySwitch) + np.log(noRecomb2) + f[self.n1+self.n2:2*self.n1+self.n2, j-1]
            term3_pop2 = np.log(noAncestrySwitch) + np.log(noRecomb1) + f[self.n1:self.n1+self.n2, j-1]

            # term1
            f[self.n1:self.n1+self.n2, j] = np.repeat(term1_pop2, self.n2)
            f[self.n1+self.n2:2*self.n1+self.n2, j] = np.repeat(term1_pop1, self.n1)
            f[self.n1:self.n1+self.n2, j] = np.apply_along_axis(np.logaddexp, 
                                                                0, np.repeat(term2_pop2, self.n2), 
                                                                f[self.n1:self.n1+self.n2, j])
            f[self.n1+self.n2:2*self.n1+self.n2, j] = np.apply_along_axis(np.logaddexp,
                                                                          0, np.repeat(term2_pop1, self.n1),
                                                                          f[self.n1+self.n2:2*self.n1+self.n2, j])
            f[self.n1:self.n1+self.n2, j] = np.apply_along_axis(np.logaddexp, 0, 
                                                                term3_pop2, f[self.n1:self.n1+self.n2, j])
            f[self.n1+self.n2:2*self.n1+self.n2, j] = np.apply_along_axis(np.logaddexp, 0,
                                                                          term3_pop1, f[self.n1+self.n2:2*self.n1+self.n2, j])
            f[:, j] = emis[j] + f[:, j]
        return f

    @jit(parallel=True)
    def backward(self, emis):
       logMiscopy, logNoMiscopy = np.log(self.miscopy), np.log(1-self.miscopy)
       b = np.zeros((2*(self.n1+self.n2), self.numSNP))
       for j in range(self.numSNP-2, -1, -1):
           noAncestrySwitch, noRecomb1, noRecomb2 = self.transition(self.D[j+1])
           # here we pivot on the first entry in the triplet (in forward matrix calculation, I pivoted on the second entry)

           # first compute terms for i != l
           term_noMiscopy_pop1 = logsumexp(np.log(1-noAncestrySwitch) + np.log(1-self.mu) + logNoMiscopy - 
                                           np.log(self.n2) + emis[j+1, 2*self.n1+self.n2:2*(self.n1+self.n2)] + 
                                           b[2*self.n1+self.n2:2*(self.n1+self.n2), j+1])
           term_Miscopy_pop1 = logsumexp(np.log(1-noAncestrySwitch) + np.log(1-self.mu) + logMiscopy -
                                         np.log(self.n1) + emis[j+1, self.n1+self.n2:2*self.n1+self.n2] +
                                         b[self.n1+self.n2:2*self.n1+self.n2, j+1])
           term_noMiscopy_pop2 = logsumexp(np.log(1-noAncestrySwitch) + np.log(self.mu) + logNoMiscopy - 
                                           np.log(self.n1) + emis[j+1, :self.n1] + b[:self.n1, j+1])
           term_Miscopy_pop2 = logsumexp(np.log(1-noAncestrySwitch) + np.log(self.mu) + logMiscopy -
                                         np.log(self.n2) + emis[j+1, self.n1:self.n1+self.n2] +
                                         b[self.n1:self.n1+self.n2, j+1])
           b[:self.n1+self.n2, j] = np.repeat(np.logaddexp(term_noMiscopy_pop1, term_Miscopy_pop1), self.n1+self.n2)
           b[self.n1+self.n2:, j] = np.repeat(np.logaddexp(term_noMiscopy_pop2, term_Miscopy_pop2), self.n1+self.n2)

           # add terms for i=l and l=m
           # the first term is shared by both i=j and i != j
           transition_pop1 = np.logaddexp(np.log(noAncestrySwitch) + np.log(1-noRecomb1) + logNoMiscopy - np.log(self.n1),
                                          np.log(1-noAncestrySwitch) + np.log(self.mu) + logNoMiscopy - np.log(self.n1))
           term_pop1 = logsumexp(transition_pop1 + emis[j+1, :self.n1] + b[:self.n1, j+1])
           b[:self.n1+self.n2, j] = np.apply_along_axis(np.logaddexp, 0,
                                                        np.repeat(term_pop1, self.n1+self.n2), b[:self.n1+self.n2, j])

           transition_pop2 = np.logaddexp(np.log(noAncestrySwitch) + np.log(1-noRecomb2) + logNoMiscopy - np.log(self.n2),
                                          np.log(1-noAncestrySwitch) + np.log(1-self.mu) + logNoMiscopy - np.log(self.n2))
           term_pop2 = logsumexp(transition_pop2 + emis[j+1, 2*self.n1+self.n2:] + b[2*self.n1+self.n2:, j+1])
           b[self.n1+self.n2:, j] = np.apply_along_axis(np.logaddexp, 0,
                                                        np.repeat(term_pop2, self.n1+self.n2), b[self.n1+self.n2:, j])

           # the next term is exclusively for i=j
           noEventTransition_pop1 = np.log(noAncestrySwitch) + np.log(noRecomb1)
           noEventTransition_pop2 = np.log(noAncestrySwitch) + np.log(noRecomb2)
           term_pop1 = noEventTransition_pop1 + emis[j+1, :self.n1] + b[:self.n1, j+1]
           term_pop2 = noEventTransition_pop2 + emis[j+1, 2*self.n1+self.n2:] + b[2*self.n1+self.n2:, j+1]
           b[:self.n1, j] = np.apply_along_axis(np.logaddexp, 0, term_pop1, b[:self.n1, j])
           b[2*self.n1+self.n2:, j] = np.apply_along_axis(np.logaddexp, 0, term_pop2, b[2*self.n1+self.n2:, j])

           # we add terms for l=i and m != l
           transition_pop1 = np.logaddexp(np.log(noAncestrySwitch) + np.log(1-noRecomb1) + logMiscopy - np.log(self.n2),
                                          np.log(1-noAncestrySwitch) + np.log(self.mu) + logMiscopy - np.log(self.n2))
           term_pop1 = logsumexp(transition_pop1 + emis[j+1, self.n1:self.n1+self.n2] + b[self.n1:self.n1+self.n2, j+1])
           b[:self.n1+self.n2, j] = np.apply_along_axis(np.logaddexp, 0, 
                                                        np.repeat(term_pop1, self.n1+self.n2), b[:self.n1+self.n2, j])

           transition_pop2 = np.logaddexp(np.log(noAncestrySwitch) + np.log(1-noRecomb2) + logMiscopy - np.log(self.n1),
                                          np.log(1-noAncestrySwitch) + np.log(1-self.mu) + logMiscopy - np.log(self.n1))
           term_pop2 = logsumexp(transition_pop2 + emis[j+1, self.n1+self.n2:2*self.n1+self.n2] + b[self.n1+self.n2:2*self.n1+self.n2, j+1])
           b[self.n1+self.n2:, j] = np.apply_along_axis(np.logaddexp, 0, 
                                                        np.repeat(term_pop2, self.n1+self.n2), b[self.n1+self.n2:, j])

           # the final term is for l=i, j != i and l != m
           term_pop1 = noEventTransition_pop1 + emis[j+1, self.n1:self.n1+self.n2] + b[self.n1:self.n1+self.n2, j+1]
           b[self.n1:self.n1+self.n2, j] = np.apply_along_axis(np.logaddexp, 0, 
                                                               term_pop1, b[self.n1:self.n1+self.n2, j])
           term_pop2 = noEventTransition_pop2 + emis[j+1, self.n1+self.n2:2*self.n1+self.n2] + b[self.n1+self.n2:2*self.n1+self.n2, j+1]
           b[self.n1+self.n2:2*self.n1+self.n2] = np.apply_along_axis(np.logaddexp, 0,
                                                                      term_pop2, b[self.n1+self.n2:2*self.n1+self.n2, j])
       return b



    
    @jit(parallel=True)
    def posterior(self, f, b):
        # posterior decoding
        pass
    

    def decode(self, obs):
        # infer hidden state of each SNP sites in the given haplotype
        # state[j] = 0 means site j was most likely copied from population 1 
        # and state[j] = 1 means site j was most likely copies from population 2

        start = time.time()
        emis = self.emissionALL(obs)
        n1, n2 = self.n1, self.n2
        ncol = self.numSNP
        f = self.forward(emis)
        b = self.backward(emis)
        end = time.time()
        print(f'uncached version takes time {end-start}')
        print(f'forward probability:{logsumexp(f[:,-1])}')
        print(f'backward probability:{logsumexp(self.initial + emis[0] + b[:,0])}')
        return 1,2
        #post_pop1, post_pop2 = self.posterior(f,b)
        #return post_pop1, post_pop2

