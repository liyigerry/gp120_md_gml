Source [data](./data/) and [scripts](./code/) for the paper "Deciphering the sequence variation and structural dynamics of envelope glycoprotein gp120 in HIV neutralization phenotype by molecular dynamics simulation and graph machine learning"

This paper aims to decipher the roles of rapid sequence variability and significant structural dynamics of envelope glycoprotein gp120 in HIV neutralization phenotype. 45 prefusion gp120 from different HIV strains belong to three tiers of sensitive, moderate, and resistant neutralization phenotype are structurally modeled by homology modeling and investigated by molecular dynamics simulations and graph machine learning.

### Data
1. All HIV [sequences](./data/seq/) with the neutralization- [sensitive](./data/seq/sensitive.fasta), [moderate](./data/seq/moderate.fasta), and [resistant](./data/seq/resistant.fasta) neutralization phenotype based on the experimental assessment of HIV neutralization phenotype with a broad range of genetic and geographic diversity were obtained from the UniProtKB database (http://www.uniprot.org).
2. All gp120 [structural models](./data/stru/) from different HIV strains with the neutralization- [sensitive](./data/stru/sensitive), [moderate](./data/stru/moderate/), and [resistant](./data/stru/resistant/) neutralization phenotype were randomly selected and constructed by homology modeling.
3. 45 prefusion gp120 from different HIV strains with the neutralization- [sensitive](./data/traj/sensitive), [moderate](./data/traj/moderate/), and [resistant](./data/traj/resistant/) neutralization phenotype are investigated by molecular dynamics simulations (only [Ca trajectories](./data/traj) are uploaded).

### Scripts
1. structural deviations: root mean square deviation [RMSD](./code/rmsd.ipynb) and radius gyration [Rg](./code/rg.ipynb) of the hydrophobic core.
2. population distribution: free energy landscapes [FEL](./code/fel.ipynb).
3. conformational flexibility: root-mean-square fluctuation [RMSF](./code/rmsf.ipynb).
4. [Model selection](./code/models.ipynb) from the Gated Recurrent Unit (GRU), Graph Convolution Network (GCN) and Graph Isomorphism Network (GIN).
5. Graph Isomorphism Network (GIN) [training](./code/gin_dyn.ipynb) on molecular dynamics simulations.
6. Graph Isomorphism Network (GIN) with attention mechanism [GIN_ATT](./code/gin_att.ipynb).