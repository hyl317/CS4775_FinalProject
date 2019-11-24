# calibrate posterior probability obtained by the hmm model
# bin probability by 0.05 (thus 20 bins in total)

import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
import gzip
import io

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
        locs = np.where(np.logical_and(posteriorMatrix >= i*bin, posteriorMatrix < (i+1)*bin))
        meanPosterior = np.mean(posteriorMatrix[locs])
        empiricalFreq = np.sum(ancestryMatrix[locs] == 0)/len(locs[0])
        post.append(meanPosterior)
        empirical.append(empiricalFreq)
    return post, empirical



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
    labels = ['t=6', 't=20', 't=50', 't=100']
    plt.figure()
    plt.xlabel('Posterior probability of originating from CEU')
    plt.ylabel('Empirical probability of originating from CEU')
    plt.plot([0,1],[0,1], color='black',linewidth=2)
    for decodeFile, refAncestryFile, c, label in zip(decodeFileList, refAncestryFileList, color, labels):
        averagePost, empiricalFreq = calibrate(decodeFile, refAncestryFile, bin=binSize)
        plt.plot(averagePost, empiricalFreq, color=c, label=label)

    plt.legend(loc='upper left', fontsize='large')
    plt.savefig('calibrate.png')


if __name__ == '__main__':
    main()
