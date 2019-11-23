# plot the posterior probability of belonging to population 1(specified via -p1) along a single chromosome

import argparse
import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

def referenceprovided(decode, reference, dir):
    with gzip.open(decode) as posterior:
        with io.BufferedReader(posterior) as posterior, open(reference, 'r') as ref:
            pop1, pop2 = ref.readline().strip().split(' ')
            count = 1
            for post in posterior:
                referenceAncestry = ref.readline().strip().split(' ')
                post = post.decode('latin1')
                post = post.strip().split('\t')
                plot(post, pop1, count, dir, referenceAncestry)
                count += 1
                break


def noReference(decode):
    pass



def plot(posterior, pop1, count, dir, true_Ancestry=None):
    posterior = list(map(float, posterior))
    plt.plot(np.arange(len(posterior)), posterior, linewidth=3, color='black')
    plt.xlabel('SNP site along the chromosome')
    plt.ylabel(f'Posterior Probability of Belonging to {pop1}')
    plt.ylim(0, 1.1)

    pop1_interval = []
    prev_pop, prev_snp = None, 0
    for switch in true_Ancestry:
        pop, curr = switch.split(':')
        pop, curr = int(pop), int(curr)
        if pop == 0:
            if prev_pop == None:
                pop1_interval.append((0, curr))
            elif prev_pop == 0:
                temp = pop1_interval[-1]
                pop1_interval[-1] = (temp[0], curr)
            else:
                pop1_interval.append(prev+1, curr)
        prev_pop, prev_snp = pop, curr

    ax = plt.gca()
    for interval in pop1_interval:
        rect = patches.Rectangle((interval[0], 0), interval[1]-interval[0]+1, 1, alpha=0.5, color="grey")
        ax.add_patch(rect)

    plt.savefig(f'./{dir}/posterior.vs.ref.{count}.png')




def main():
    parser = argparse.ArgumentParser(description='Visualize posterior probability.')
    parser.add_argument('-p', action="store", dest="p", type=str, required=True, help='path to the zipped posterior decoding file')
    parser.add_argument('-r', action="store", dest="r", type=str, help='path to the true ancestry.[OPTIONAL]')
    parser.add_argument('-o', action="store", dest="o", type=str, required=True, help='path to output directory where images will saved')
    args = parser.parse_args()

    if args.r != None:
        referenceprovided(args.p, args.r, args.o)
    else:
        noReference(args.p)



if __name__ == '__main__':
    main()


