# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 07:32:01 2020

Author: Ruby Li

This script includes motif logos generation

"""
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from weblogo import *

# Read in mhc_ligand_full.csv and select data

mhc_full = pd.read_csv('mhc_ligand_full.csv')
mhc_full = mhc_full[mhc_full['Date']>2000]
mhc_full = mhc_full[mhc_full['Object Type'] == 'Linear peptide']
mhc_full = mhc_full[mhc_full['Units'] == 'nM']
mhc_full = mhc_full[mhc_full['Quantitative measurement']<500]

# obtain epitope sequences and save them into a file
def IRI2fa(IRI):
    first_line = IRI.replace('http://www.iedb.org/epitope/', '>IEDB Epitope ')
    res = requests.get(IRI)
    soup = BeautifulSoup(res.text, 'lxml')
    second_line = soup.title.text
    second_line = second_line.replace(' epitope - Immune Epitope Database (IEDB)', '')
    return first_line, second_line

# get epitope sequence
sequence_list = []
for i in range(31357, mhc_full.shape[0]):
    link = mhc_full['Epitope IRI'].iloc[i]
    res = requests.get(link)
    soup = BeautifulSoup(res.text, 'lxml')
    seq = soup.title.text
    sequence_list.append(seq.replace(' epitope - Immune Epitope Database (IEDB)', ''))
    print(str(i) + ' ' + seq.replace(' epitope - Immune Epitope Database (IEDB)', ''))

# generate fa files
file = open('IEDBs\hla0301.txt', 'w')
for i in range(hla0301.shape[0]):
    first_line, second_line = IRI2fa(hla0301['Epitope IRI'].iloc[i])
    file.write(first_line + '\n')
    file.write(second_line + '\n')
    print(i)
file.close()

# generate motif logos
fin = open('hla0301.fa')
seqs = read_seq_data(fin)
logodata = LogoData.from_seqs(seqs)
logooptions = LogoOptions()
logooptions.title = "HLA0301"
logoformat = LogoFormat(logodata, logooptions)
eps = eps_formatter(logodata, logoformat)
