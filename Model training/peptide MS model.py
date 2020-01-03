# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 07:23:43 2020

Author: Ruby Li

This script includes codes for peptide MS model

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

####################
# Keras model code #
####################


peptide_in = Input(shape=(len_peptide,), dtype='uint8', name='peptide')
peptide_onehot = Lambda(lambda x: K.one_hot(x, len_AA_dict),
                        output_shape=(len_peptide, len_AA_dict, ),
                        name='peptide_onehot')(peptide_in)
peptide_flatten = Flatten(name='peptide_flatten')(peptide_onehot)
peptide_dense = Dense(256,
                      activation='relu',
                      name='peptide_dense')(peptide_flatten)
peptide_out = Dense(n_HLAs - 1,
                    activation='linear',
                    name='peptide_out')(peptide_dense)

model_out = Lambda(lambda x: K.sigmoid(x),
                   output_shape=(n_HLAs - 1, ),
                   name='model_out')(peptide_out)


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
model_out = Dot(-1, name='hla_deconv')([model_out, hla_out])


model = Model(inputs=[peptide_in,
                      hla_in
                      ],
              outputs=model_out)
model.compile(optimizer=Adam(), loss='binary_crossentropy')

early_stopping = EarlyStopping(monitor='val_loss', mode = 'min',verbose = 1)

model_inputs = {'peptide': peptide_encode(peptide, maxlen=len_peptide),
                'hla_onehot': hla_encode(hla_onehot)}

X_val = {'peptide':peptide_encode(val_peptide, maxlen=len_peptide),
         'hla_onehot':hla_encode(val_onehot)}
y_val = val_labels


model.fit(model_inputs, labels, validation_data = (X_val, y_val), epochs=100, verbose = True, callbacks = [early_stopping])
