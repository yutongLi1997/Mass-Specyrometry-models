# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 07:05:08 2020

Author: Ruby Li

This script includes obtaining family id, sample id, HLAs and tpm

"""
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re

# Assign HLAs
def get_hla(sample_name, hla_data, df):
    sample_id = sample_name.lower().replace(' ','_')
    tmp = hla_data[hla_data['paper_sample_id'] == sample_id]
    df['A1'] = tmp['A1'].iloc[0]
    df['B1'] = tmp['B1'].iloc[0]
    df['C1'] = tmp['C1'].iloc[0]
    df['A2'] = tmp['A2'].iloc[0]
    df['B2'] = tmp['B2'].iloc[0]
    df['C2'] = tmp['C2'].iloc[0]
    return df

# Obtain family ids
def get_familyid(soup, uniprot_id):
    trs = soup.find_all('tr')[34:]
    for tr in trs:
        for item in tr.find_all('td'):
            if uniprot_id in item:
                tmp = str(tr)
                family_id = re.findall('\((PTHR.*?)\)', tmp)[0]
                return family_id
            else:
                family_id = ''
    return family_id

# Compute tpm
def get_geneid(file_name):
    tree = et.ElementTree(file = FOLDER + file_name)
    root = tree.getroot()
    gene_id= []
    for child in root:
        for grchild in child:
            if grchild.tag == '{http://uniprot.org/uniprot}dbReference':
                if(grchild.attrib['type'] == 'Ensembl'):
                    gene_id.append(grchild.attrib['id'])
        return gene_id
    
def compute_tpm(geneids, tpm_data, sample_name):
    tpm = 0.0
    for gene_id in geneids: 
        transcript = tpm_data[tpm_data['transcript_id'].str.contains(gene_id)]
        transcript['isDuplicate'] = transcript.duplicated(sample_name)
        transcript = transcript[transcript['isDuplicate'] == False]
        if transcript.empty:
            tpm += 0.0
        elif transcript.shape[0] == 1:
            tpm += float(transcript[sample_name])
        else:
            for i in range(transcript.shape[0]):
                tpm += float(transcript[sample_name].iloc[i])
    return tpm

