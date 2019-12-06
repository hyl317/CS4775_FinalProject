import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn

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
    parser = argparse.ArgumentParser(description='k-means clustering using sklearn package.')
    parser.add_argument('-e', action="store", dest="e", type=str, required=True, 
                        help='eigenstrat SNP file')
    parser.add_argument('-n', action="store", dest="n", type=int, required=True, 
                        help='number of clusters')
    args = parser.parse_args()

    X = readEigenstrat(args.e).T
    km = KMeans(n_clusters=args.n, init='k-means++')
    Y = km.fit_predict(X)
    pca = PCA(n_components=2)
    pc = pca.fit_transform(X)
    pc_df = pd.DataFrame(data=pc, columns=['PC1','PC2'])
    pc_df['cluster'] = Y

    snsplot = sns.lmplot(x='PC1', y='PC2', data=pc_df, fit_reg=False, hue='cluster', 
               legend=True, scatter_kws={'s':50})
    snsplot.savefig('pca.png')

if __name__ == '__main__':
    main()
