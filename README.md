# BTRY4840/CS4775 FinalProject

This is a implementation of the haploid version of [Hapmix](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000519). Hapmix is a HMM based local ancestry inference tool for two-way single admixture event.

## Dependency
The following python libraries are required. And make sure hapmix.py and hmm_fast.py are in the same directory.

- numpy
- scipy
- numba


## Usage
The main program is hapmix.py. Use `python hapmix.py -h`, you can view several necessary input files to run the program along with a short description.

```
python hapmix.py -h
usage: hapmix.py [-h] -p1 P1 -p2 P2 -a A -m M [-mu MU] [-t T]
                 [--miscopy MISCOPY] [--mis]

Local Ancestry Inference as implemented in Hapmix.

optional arguments:
  -h, --help         show this help message and exit
  -p1 P1             eigenstrat SNP file for ancestral population 1 (eg. CEU)
  -p2 P2             eigenstrat SNP file for ancestral population 2 (eg. YRI)
  -a A               eigenstrat SNP file for the admixed population.
  -m M               genetic map. i.e, the .snp file used in admixture
                     simulation.
  -mu MU             percentage of genetic composition of population 1 in the
                     admixed population. Default=0.2
  -t T               Time (in numbers of generation) to the admixture event.
                     Default=5
  --miscopy MISCOPY  miscopying probability. Default is 0.05.
  --mis              If this flag is asserted, then miscopy will be allowed.
```

The eigenstrat, .snp file format specification and examples can be found in [admix-simu](https://github.com/williamslab/admix-simu).

## Output Files
Upon completion, the hapmix.py will output two files, one called "decode.txt", and the other called "raw.posterior.gz"(it's by default zipped).

The decode.txt file describes inferred ancestry switches. It contains one line per haplotype. Each line takes the format \[population index\]:\[marker index\]. Population index 0 corresponds to the ancestral population given via -p1 and index 1 corresponds to the ancestral population given via -p2. The marker is indexed from zero, inclusive. For example, 
```
1:66606 0:87954
```
The above line says that from markers from 0 up to and inclusing 66606 are inferred to have ancestry from -p2, and markers from 66607 up to and including 87954 are inferred to have ancestry from -p1.

The raw.posterior.gz file contains posterior probability, one line per haplotype, one column for SNP site, of one SNP having ancestry from -p1.

