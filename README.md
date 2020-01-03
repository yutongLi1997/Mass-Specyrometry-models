# Mass-Specyrometry-models
Deep learning using tumor HLA peptide mass spectrometry datasets improves neoantigen identification
This project is to repeat the work from paper: [Deep learning using tumor HlA peptide mass spectrometry datasets improves neoantigen identification.](https://www.nature.com/articles/nbt.4313) 
## Packages
* Python 3.7
* Keras
* BeautifulSoup
* Theano
* MHCflurry
## Pipeline
![The pipeline of this project is showed as follow:](https://github.com/yutongLi1997/Mass-Specyrometry-models/blob/master/pipeline.png)
#### The Identification of Neoantigen-reactive T Cells in Cancer Patients is not included in the code.
## Datasets
All the data used for MS model is available on the [online version of the paper.](https://www.nature.com/articles/nbt.4313#Sec33)
The data for generating binding affinity logos is available on Immune Epitope Database ([IEDB])(https://www.iedb.org/database_export_v3.php)
## Features
**Ppetide Sequences:** The peptide sequences data is available in MS dataset
**Flanking Sequences:** This can be done by taking the N-terminal and C-terminal flanking sequence of the corresponding peptide
**HLA alleles**: The HLA alleles data is available in Supplementary Dataset 1.
**Sample id**: The sample id data is available in Supplementart Dataset 1.
**TPM**: This can be done by taking the logarithm of per-peptide TPM.
**Family id**: The family id can be obtained by requesting [PantherDB](http://pantherdb.org/geneListAnalysis.do).
## Data Pre-Processing
1. Remove all the peptides whose lengths are not in the range of 8-11 mer
2. Vectorize the peptides and flanking sequences by a one-hot scheme
3. Compute per-peptide TPM as the sum of per-isoform TPM
# Full MS model
The full presentation model has the following functional form:
<img src="http://chart.googleapis.com/chart?cht=tx&chl= Pr(peptide i presented) = \sum_(k = 1)^{m}a_{k}^{i}*Pr(peptide i presented by allele k)" style="border:none;">

