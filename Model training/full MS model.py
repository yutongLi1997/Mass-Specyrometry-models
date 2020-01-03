# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 07:13:52 2020

Author: Ruby Li

This script includes codes for full MS model

"""
import numpy as np
import theano
import keras
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Add, Dense, Dot, Flatten, Input, Lambda, RepeatVector
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math

################################
# Model settings and constants #
################################


len_peptide = 11  # the padded peptide length, i.e., the max peptide length
len_flanking = 10  # 5 n-terminal and 5 c-terminal flanking amino acids
hla_per_sample = 6


AA_dict = {'A': 1,
           'C': 2,
           'D': 3,
           'E': 4,
           'F': 5,
           'G': 6,
           'H': 7,
           'I': 8,
           'K': 9,
           'L': 10,
           'M': 11,
           'N': 12,
           'P': 13,
           'Q': 14,
           'R': 15,
           'S': 16,
           'T': 17,
           'V': 18,
           'W': 19,
           'Y': 20,
           'Z': 0}  # 'Z' is a padding variable

def remove_dupilcated_hlas(hla_list):
    non_duplicates = []
    for hla in hla_list:
        if not hla in non_duplicates:
            non_duplicates.append(hla)
    return non_duplicates
HLAs = remove_dupilcated_hlas(HLAs)

len_AA_dict = len(AA_dict)
n_HLAs = len(HLAs) + 1  # 0 is for a blank allele (for homozygotes)
n_sample_id = len(Samples)
n_protein_family = len(Proteins)

########################################
# Input data encoding helper functions #
########################################


def category_encode(data, categories):
    '''Convert categorical data to a numberic representation.

    Parameters
    ----------
    data : list
        Cateogorical data to be converted.
    categories : list
        An ordered list of the category tokens.

    Returns
    -------
    
    encoded: np.ndarray
        A numerically encoded representation of the input data.
    '''
    if isinstance(data, str):
        data = [data]
    if isinstance(data, np.ndarray):
        data = data.tolist()
    encoded = []
    for datum in data:
        if datum not in categories:
            raise ValueError('Category not found!: %s' % datum)
        encoded.append(categories.index(datum))
    return np.array(encoded)


def hla_encode(alleles, hla_per_sample=hla_per_sample, HLAs=HLAs):
    '''Convert the HLAs of a sample(s) to a zero-padded (for homozygotes)
    numeric representation.

    Parameters
    ----------
    alleles: list
        A list of alleles (from HLAs) of length 1-hla_per_sample, or 
        a list of lists of alleles.
    hla_per_sample: int
        The maximum number of unique alleles per sample (typically 6).
    HLAs: list
        An alphabet of HLA alleles.
    '''
    if isinstance(alleles, np.ndarray):
        alleles = alleles.tolist()
    type_check = [isinstance(sample, list) for sample in alleles]
    if any(type_check):
        assert all(type_check), \
            'Must provide either a list of alleles or a list of allele lists!'
    else:
        alleles = [alleles]
    onehots = []
    for sample in alleles:
        onehot = category_encode(sample, HLAs)
        onehot = [code + 1 for code in onehot]
        onehot = [0] * (hla_per_sample - len(onehot)) + onehot
        onehots.append(onehot)

    return np.array(onehots)


def peptide_encode(peptides, maxlen=None, AA_dict=AA_dict):
    '''Convert peptide amino acid sequence to one-hot encoding,
    optionally left padded with zeros to maxlen.

    The letter 'Z' is interpreted as the padding character and
    is assigned a value of zero.

    e.g. encode('SIINFEKL', maxlen=12)
             := [16,  8,  8, 12,  0,  0,  0,  0,  5,  4,  9, 10]

    Parameters
    ----------
    peptides : list-like of strings over the amino acid alphabet
        Peptides
    maxlen : int, default None
        Pad peptides to this maximum length. If maxlen is None,
        maxlen is set to the length of the first peptide.

    Returns
    -------
    onehot : 2D np.array of np.uint8's over the alphabet [0, 20]
        One-hot encoded and padded peptides. Note that 0 is padding, 1 is
        Alanine, and 20 is Valine.
    '''
    if isinstance(peptides, str):
        peptides = [peptides]
    num_peptides = len(peptides)
    if maxlen is None:
        maxlen = max(map(len, peptides))
    onehot = np.zeros((num_peptides, maxlen), dtype=np.uint8)
    for i, peptide in enumerate(peptides):
        if len(peptide) > maxlen:
            msg = 'Peptide %s has length %d > maxlen = %d.'
            raise ValueError(msg % (peptide, len(peptide), maxlen))
        o = list(map(lambda x: AA_dict[x], peptide))
        k = len(o)
        o = o[:k // 2] + [0] * (maxlen - k) + o[k // 2:]
        if len(o) != maxlen:
            msg = 'Peptide %s has length %d < maxlen = %d, but pad is "none".'
            raise ValueError(msg % (peptide, len(peptide), maxlen))
        onehot[i, :] = o
    return np.array(onehot)

####################
# Keras model code #
####################

# peptide NN
peptide_in = Input(shape=(len_peptide,), dtype='uint8', name='peptide') # input layer
peptide_onehot = Lambda(lambda x: K.one_hot(x, len_AA_dict),
                        output_shape=(len_peptide, len_AA_dict, ),
                        name='peptide_onehot')(peptide_in)
peptide_flatten = Flatten(name='peptide_flatten')(peptide_onehot)
peptide_dense = Dense(256, # hidden layer
                      activation='relu',
                      name='peptide_dense')(peptide_flatten)
peptide_out = Dense(n_HLAs - 1, # output layer
                    activation='linear',
                    name='peptide_out')(peptide_dense)

# flanking NN
flanking_in = Input(shape=(len_flanking,), dtype='uint8', name='flanking')
flanking_onehot = Lambda(lambda x: K.one_hot(x, len_AA_dict),
                         output_shape=(len_flanking, len_AA_dict, ),
                         name='flanking_onehot')(flanking_in)
flanking_flatten = Flatten(name='flanking_flatten')(flanking_onehot)
flanking_dense = Dense(32,
                       activation='relu',
                       name='flanking_dense')(flanking_flatten)
flanking_out = Dense(1,
                     activation='linear',
                     name='fanking_out')(flanking_dense)

# tpm nn
log10_tpm_in = Input(shape=(1,), dtype='float32', name='log10_tpm')
log10_tpm_dense = Dense(16,
                        activation='relu',
                        name='log10_tpm_dense')(log10_tpm_in)
log10_tpm_out = Dense(1,
                      activation='linear',
                      name='log10_tpm_out')(log10_tpm_dense)


sample_id_in = Input(shape=(1,), dtype='int32', name='sample_ids')
sample_id_embed = Embedding(n_sample_id,
                            1,
                            input_length=1,
                            name='sample_id_embed')(sample_id_in)
sample_id_flatten = Flatten(name='sample_id_flatten')(sample_id_embed)


protein_family_in = Input(shape=(1,), dtype='int32', name='protein_family')
protein_family_embed = Embedding(n_protein_family,
                                 1,
                                 input_length=1)(protein_family_in)
protein_family_flatten = Flatten(name='protein_family_flatten')(protein_family_embed)


noninteract = Add(name='noninteract_add')([flanking_out,
                                           log10_tpm_out,
                                           sample_id_flatten,
                                           protein_family_flatten])
noninteract = RepeatVector(n_HLAs - 1,
                           name='noninteract_repeat')(noninteract)
noninteract = Flatten(name='noninteract_tile_flatten')(noninteract)

# probability of peptide i presented by allele i
model_out = Add(name='add_interact_noninteract')([peptide_out, noninteract])
model_out = Lambda(lambda x: K.sigmoid(x),
                   output_shape=(n_HLAs - 1, ),
                   name='model_out')(model_out)

# hla NN
hla_in = Input(shape=(hla_per_sample,), dtype='uint16', name='hla_onehot')
hla_embed = Embedding(n_HLAs,
                      n_HLAs - 1,
                      input_length=hla_per_sample,
                      trainable=False,
                      weights=[np.eye(n_HLAs)[:, 1:]],
                      name='hla_embed')(hla_in)
hla_out = Lambda(lambda x: K.sum(x, axis=1, keepdims=False),
                 output_shape=(n_HLAs - 1, ),
                 name='hla_out')(hla_embed)

# probability of peptide i presented
model_out = Dot(-1, name='hla_deconv')([model_out, hla_out])


model = Model(inputs=[peptide_in,
                      flanking_in,
                      hla_in,
                      protein_family_in,
                      sample_id_in,
                      log10_tpm_in
                      ],
              outputs=model_out)
model.compile(optimizer=Adam(), loss='binary_crossentropy')

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', mode = 'min',verbose = 1)

# model training
model_inputs = {'peptide': peptide_encode(peptide, maxlen=len_peptide),
                'flanking': peptide_encode(flanking),
                'hla_onehot': hla_encode(hla_onehot),
                'protein_family': category_encode(protein_family, Proteins),
                'sample_ids': category_encode(sample_ids, Samples),
                'log10_tpm': log10_tpm}

X_val = {'peptide':peptide_encode(val_peptide, maxlen=len_peptide),
         'flanking': peptide_encode(val_flanking),
         'hla_onehot':hla_encode(val_onehot),
         'protein_family': category_encode(val_protein_family, Proteins),
         'sample_ids':category_encode(val_sample_ids, Samples),
         'log10_tpm': val_log10_tpm}
y_val = val_labels

model.fit(model_inputs, labels, validation_data = (X_val, y_val), epochs=100, verbose = True, callbacks = [early_stopping])


