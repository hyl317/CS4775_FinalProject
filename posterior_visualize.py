# plot the posterior probability of belonging to population 1(specified via -p1) along a single chromosome
# If true ancestry is provided(in a simulation setting), regions trully belonging to population 1 are shaded
# note in admix-simu, the population 1 is indexed with 0

import argparse
import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def referenceprovided(decode, reference):
    with gzip.open(decode,'rb') as posterior:
        with open(reference, 'r') as ref:
            pop1, pop2 = ref.readline().strip().split(' ')
            count = 1
            for post in posterior:
                referenceAncestry = ref.readline().strip().split(' ')
                plot(post.strip().split('\t'), pop1, count, referenceAncestry)
                count++
                break


def noReference(decode):
    pass



def plot(posterior, pop1, count, true_Ancestry=None):
    posterior = list(map(int, posterior))
    plt.plot(np.arange(len(posterior)), posterior, linewidth=5)
    plt.xlabel('SNP site along the chromosome')
    plt.ylabel(f'Posterior Probability of Belonging to {pop1}')
    plt.ylim(0, 1.1)

    ax = plt.gca()
    prev = 0
    for switch in true_Ancestry:
        pop, curr = switch.split(':')
        pop, curr = int(pop), int(curr)
        if pop == 0:
            rect = patches.Rectangle((prev, 0), curr, 1, alpha=0.5, color="grey")
            ax.add_patch(rect)

    plt.savefig(f'posterior.vs.ref.{count}.png')




def main():
    parser = argparse.ArgumentParser(description='Visualize posterior probability.')
    parser.add_argument('-p', action="store", dest="p", type=str, required=True, help='path to the zipped posterior decoding file')
    parser.add_argument('-r', action="store", dest="r", type=str, help='path to the true ancestry.[OPTIONAL]')
    parser.add_argument('-o', action="store", dest="o", type=str, required=True, help='path to output directory where images will saved')
    args = parser.parse_args()

    if args.r != None:
        referenceprovided(args.p, args.r)
    else:
        noReference(args.p)



if __name__ == '__main__':
    main()


