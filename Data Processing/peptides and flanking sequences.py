# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Ruby Li

This script includes how to raw peptide data cleaning and flanking sequences 
generation

"""
import pandas as pd
import numpy as np
import xml.etree.cElementTree as et
import requests
import os

# Select peptides whose length is between 8-11
data = data.loc[(data.peptide.str.len() >= 8) & (data.peptide.str.len() <= 11)]

# Obtain data from Uniprot
FOLDER = 'uniprots\\'

# Obtain xml files from Uniprot
def get_files(resource):
    bol = False
    # Uniprot webservice
    sequence_file = requests.get('http://www.uniprot.org/uniprot/' + resource)

    # If response is empty
    if len(sequence_file.text) == 0:
        print('not available in .xml or does not exist')
        

    # http not 200
    elif sequence_file.status_code != 200:
        print('http error ' + str(sequence_file.status_code))

    # If response is html, then it's invalid
    else:
        html = False
        for line in sequence_file.iter_lines():
            if '<!DOCTYPE html' in line.decode():
                print('not available in .' + output_format + ' or does not exist')
                html = True

        with open(FOLDER + resource, "wb") as file_name:
            [file_name.write(line + '\n'.encode()) for line in sequence_file.iter_lines()]
            bol = True
#     return bol

# get one protein sequence
def get_protein_seq(file_name):
    tree = et.ElementTree(file = FOLDER + file_name)
    root = tree.getroot()
    for child in root:
        for grchild in child:
            if grchild.tag == '{http://uniprot.org/uniprot}sequence':
                seq = grchild.text.strip()
                return seq

# get all protein sequences
def get_all_protseq(df):
    seq_list = []
    for i in range(df.shape[0]):
        resource = df['protein id'].iloc[i] + '.xml'
#         bol = get_files(resource)
#         print(resource)
        if(os.path.exists('uniprots\\'+resource)):
            seq = get_protein_seq(resource)
        else:
            seq = ''
        seq_list.append(seq)
#         if(i%100 == 0):
#             print('%s sequences have been extracted' %(i))
    seq_df = pd.DataFrame(seq_list)
    new_df = pd.concat([df['protein id'], seq_df, df['sequence'], df[['flanking aa\n']]], axis = 1)
    new_df.columns = ['protein id','sequence', 'peptide', 'flanking aa']
    return new_df

# Compute flanking sequence
def compute_flanking_sequence(df):
    flanking_sequence_list = []
    for i in range(df.shape[0]):
#         print(i)
        s = df['sequence'].iloc[i]
        pp = df['peptide'].iloc[i]
        f = df['flanking aa'].iloc[i].replace('\n','')
        fs = ''
        for j in range(len(s)):
#             print(j)
            if (j+5+len(pp) == len(s)-1):
                fs = ''
                flanking_sequence_list.append(fs)
                break
            if(s[j+4] == f[0]) and (s[j+5:j+5+len(pp)] == pp):
                fs+=s[j:j+5]
                fs+=s[j+5+len(pp):j+10+len(pp)]
                flanking_sequence_list.append(fs)
#                 print(fs)
                break
    df['Flanking Sequence'] = pd.DataFrame(flanking_sequence_list)
    return df