# calibrate posterior probability obtained by the hmm model
# bin probability by 0.05 (thus 20 bins in total)

import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.stats
import gzip
import io
import math
from numba import jit
import scipy.spatial

def readDecodeFile(decodeFile):
    # return a numpy 2D array M such that
    # M[i,j] is the posterior probability of the j^th SNP site originating in population 1 (CEU by default) in sample i
    posterior = []
    with gzip.open(decodeFile) as file:
        with io.BufferedReader(file) as f:
            for line in f:
                probs = list(map(float, line.decode('latin1').strip().split('\t')))
                posterior.append(probs)
    return np.array(posterior)

def constructAncestry(ancestrySwitches):
    ancestryList = []
    prev = -1
    for switch in ancestrySwitches:
        ancestry, end = switch.split(':')
        ancestry, end = int(ancestry), int(end)
        ancestryList.extend([ancestry]*(end-prev))
        prev = end
    return ancestryList


def readAncestryFile(refAncestryFile):
    ancestry = []
    with open(refAncestryFile) as f:
        f.readline() #consume the header line
        line = f.readline()
        while line:
            switch = line.strip().split(' ')
            ancestry.append(constructAncestry(switch))
            line = f.readline()
    return np.array(ancestry)

def hammingDis(posteriorMatrix, ancestryMatrix):
    numSample = posteriorMatrix.shape[0]
    dist = np.zeros(numSample)
    for i in range(numSample):
        decode = [0 if prob > 0.9 else 1 for prob in posteriorMatrix[i]]
        dist[i] = scipy.spatial.distance.hamming(decode, ancestryMatrix[i])
    return dist

@jit
def rsquared(posteriorMatrix, ancestryMatrix):
    # for each sample, calculte the r^2 between its posterior probability and its true ancestry
    numSample = posteriorMatrix.shape[0]
    r2 = np.zeros(numSample)
    for i in range(numSample):
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(posteriorMatrix[i], 1-ancestryMatrix[i])
        r2[i] = r_value if not np.all(ancestryMatrix[i] == np.mean(ancestryMatrix[i])) else 1
        if r_value < 0.1:
            print('suspiciously low correlation coefficient. Might be caused by all y values being the same. Please check further!')
            print(1-ancestryMatrix[i])
            print(posteriorMatrix[i])
        #if r2[i] < 0.7:
        #    print('maybe sth is wrong')
        #    print(i)
        #    print(1-ancestryMatrix[i])
        #    print(posteriorMatrix[i])
    return r2


def calibrate(decodeFile, refAncestryFile, bin=0.05):
    # return two vectors
    # the first vector is the posterior probability binned at bin (averaged over all posterior probability belonging in this bin)
    # the second vector is the empirical probability of beloning to population 1 (CEU, by default)

    posteriorMatrix, ancestryMatrix = readDecodeFile(decodeFile), readAncestryFile(refAncestryFile)
    print(posteriorMatrix.shape)
    print(ancestryMatrix.shape)
    if posteriorMatrix.shape != ancestryMatrix.shape:
        print('posterior decoding does not seem to match reference ancestry in terms of number of samples and SNP sites. Aborting...')
        sys.exit()

    numBins = int(1/bin)
    post = []
    empirical = []
    for i in range(numBins):
        # calculate mean posterior probability and empirical frequency
        locs = np.where(np.logical_and(posteriorMatrix >= i*bin, posteriorMatrix < (i+1)*bin))
        meanPosterior = np.mean(posteriorMatrix[locs])
        empiricalFreq = np.sum(ancestryMatrix[locs] == 0)/len(locs[0])
        post.append(meanPosterior)
        empirical.append(empiricalFreq)
        print(f'number of sites in bin{i}: {len(locs[0])}')

    #return post, empirical, rsquared(posteriorMatrix, ancestryMatrix)
    return post, empirical, hammingDis(posteriorMatrix, ancestryMatrix)


def main():
    parser = argparse.ArgumentParser(description='Calibrate Probability. Assume files are provided in the order T=6,20,50,100.')
    parser.add_argument('-p', action="store", dest="p", type=str, required=True, help='path to the zipped posterior decoding file. Multiple files should be delimited by comma.')
    parser.add_argument('-r', action="store", dest="r", type=str, required=True, help='path to the true ancestry. Multiple files should be delimited by comma. Must be in the same order as in -p argument.')
    args = parser.parse_args()

    decodeFileList, refAncestryFileList = args.p.split(','), args.r.split(',')
    if len(decodeFileList) != len(refAncestryFileList) or len(decodeFileList) != 4:
        print('Unequal number of decoding files and reference ancestry files. Exitting...')
        sys.exit()

    binSize = 0.05
    color = ['r', 'm', 'g', 'b']
    labels = ['t=30', 't=50', 't=75', 't=100']
    ham = []
    plt.figure()
    plt.xlabel('Posterior probability of originating from CEU')
    plt.ylabel('Empirical probability of originating from CEU')
    plt.title('Posterior Probability Calibration for Hapmix Miscopy=0.05')
    plt.plot([0,1],[0,1], color='black',linewidth=2)
    for decodeFile, refAncestryFile, c, label in zip(decodeFileList, refAncestryFileList, color, labels):
        print(f'processing {refAncestryFile}')
        averagePost, empiricalFreq, hamDist = calibrate(decodeFile, refAncestryFile, bin=binSize)
        plt.plot(averagePost, empiricalFreq, color=c, label=label)
        ham.append(hamDist)

    plt.legend(loc='upper left', fontsize='large')
    plt.savefig('calibrate.png')

    # plot histogram of rsquared
    fig, ax = plt.subplots(2,2,figsize=(16,16))
    # here rsquared is a vector of r^2 of samples in each value of t
    fig.suptitle(f'Hamming Distance of Hapmix with Miscopy=0.05 at Various $T$', y=0.95, fontsize=24, fontweight='bold')
    for i, (hamDist, label) in enumerate(zip(ham, labels)):
        row = math.floor(i/2)
        col = i - row*2
        ax[row, col].hist(hamDist)
        ax[row, col].set_xlabel('hamming distance')
        ax[row, col].set_ylabel('count')
        ax[row, col].set_title(f'100 Samples, {label}', fontsize=20)
        ax[row, col].text(0.6, 0.8, f'$\mu=${np.mean(hamDist):.4f}', transform=ax[row, col].transAxes, fontsize=16, fontweight='bold')
        ax[row, col].text(0.6, 0.75, f'$\sigma=${np.std(hamDist):.4f}', transform=ax[row, col].transAxes, fontsize=16, fontweight='bold')

    plt.savefig('hamDist.png')

if __name__ == '__main__':
    main()
