import numpy as np
import math
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
        #self.forward = np.full((n1+n2, numSNP), np.nan)
        #self.backward = np.full((n1+n2, numSNP), np.nan)

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

    def emission(self, obs, j):
        # return log probability of emission for site j
        pop1SNP, pop2SNP = self.pop1matrix[j][:,np.newaxis], self.pop2matrix[j][:,np.newaxis]
        pop1 = np.concatenate((pop1SNP == obs[j], pop1SNP != obs[j]), axis=1)
        pop2 = np.concatenate((pop2SNP == obs[j], pop2SNP != obs[j]), axis=1)
        theta_pop1 = np.array([1-self.theta1, self.theta1])[:,np.newaxis]
        theta_pop2 = np.array([1-self.theta2, self.theta2])[:,np.newaxis]
        emission = np.concatenate((pop1@theta_pop1, pop2@theta_pop2))
        return np.log(emission)

    def forward_cache(self, obs):
        f = np.full((self.numSNP, self.n1+self.n2), np.nan)
        #initialization
        emission0 = self.emission(obs, 0)
        f[0,:] = (-math.log(self.n1+self.n2) + emission0).flatten()

        for j in range(1, self.numSNP):
            T = self.transition(self.D[j])
            f[j] = self.emission(obs, j).flatten()+logsumexp(f[j-1][:,np.newaxis]+T, axis=1)
        return f;

        

    def forward(self, obs):
        # Given the observed haplotype, compute its forward matrix
        f = np.full((self.n1+self.n2, self.numSNP), np.nan)
        # initialization
        emission0 = self.emission(obs, 0)
        f[:,0] = (-math.log(self.n1+self.n2) + emission0).flatten()
        
         # fill in forward matrix
        for j in range(1, self.numSNP):
            T = self.transition(self.D[j])
            #print(f'transition matrix:{T}')
            # using axis=1, logsumexp sum over each column of the transition matrix
            #print(f'emission:{self.emission(obs, j).flatten()}')
            #print(f'transition times previous state:{logsumexp(f[:,j-1][:,np.newaxis] + T, axis=1)}')
            f[:, j] = self.emission(obs, j).flatten() + logsumexp(f[:,j-1][:,np.newaxis] + T, axis=1)
        return f



    def backward(self, obs):
        # Given the observed haplotype, compute its backward matrix
        pass

    def decode(self, obs):
        # infer hidden state of each SNP sites in the given haplotype
        # state[j] = 0 means site j was most likely copied from population 1 
        # and state[j] = 1 means site j was most likely copies from population 2
        start1 = time.time()
        f1 = self.forward_cache(obs)
        end1= time.time()
        print(f'cached version takes time {end1-start1}')

        start2 = time.time()
        f2 = self.forward(obs)
        end2= time.time()
        print(f'uncached version takes time {end2-start2}')
        assert np.allclose(f1.T, f2)
        return 0


