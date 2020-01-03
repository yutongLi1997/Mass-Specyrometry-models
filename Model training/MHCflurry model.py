# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 07:26:12 2020

Author: Ruby Li

This script includes MHCflurry model

"""
from mhcflurry import Class1AffinityPredictor
from mhcflurry.downloads import get_path
import pandas as pd
import numpy as np

data_path = get_path('data_curated', 'curated_training_data.no_mass_spec.csv.bz2')
df = pandas.read_csv(data_path)
df = df.loc[(df.peptide.str.len() >= 8) & (df.peptide.str.len() <= 11)]

models = {}
for hla in HLAs:
    new_predictor = Class1AffinityPredictor()
    if(df.loc[df.allele == hla].shape[0]>0):
        single_allele_train_data = df.loc[df.allele == hla].sample(21, replace = True)
    else:
        models[hla] = ''
        continue
    model = new_predictor.fit_allele_specific_predictors(n_models=1,architecture_hyperparameters_list=[{"layer_sizes": [16],"max_epochs": 5,"random_negative_constant": 5,}],peptides=single_allele_train_data.peptide.values,affinities=single_allele_train_data.measurement_value.values,allele="HLA-B*57:01")
    models[hla] = model
    
binding_affinity = []
for i in range(len(test_peptide)):
    ba = float('inf')
    for hla in test_onehot[i]:
        if (hla not in models.keys()):
            if(df.loc[df.allele == hla].shape[0]>0):
                single_allele_train_data = df.loc[df.allele == hla].sample(21, replace = True)
            else:
                continue
            model = new_predictor.fit_allele_specific_predictors(n_models=1,architecture_hyperparameters_list=[{"layer_sizes": [16],"max_epochs": 5,"random_negative_constant": 5,}],peptides=single_allele_train_data.peptide.values,affinities=single_allele_train_data.measurement_value.values,allele="HLA-B*57:01")
            models[hla] = model
        if (models[hla] == ''):
            continue
        ba_tmp = models[hla][0].predict([test_peptide[i]])
        if(ba_tmp < ba):
            ba = ba_tmp
    binding_affinity.append(ba)
