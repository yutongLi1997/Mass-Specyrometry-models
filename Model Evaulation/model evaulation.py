# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 07:29:03 2020

Author: Ruby Li

This script includes model evaulation

"""

# Compute PPV and Recall
def conditions(y, pred, paras):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(pred)):
        if (y[i] == 1) and (pred[i] == y[i]):
            tp += 1
        elif (y[i] == 1) and (pred[i] != y[i]):
            tn += 1
        elif (y[i] == 0) and (pred[i] == y[i]):
            fp += 1
        elif(y[i] == 0) and (pred[i] != y[i]):
            fn += 1
    if (paras == 'Recall'):
        if (tp == 0) and (fn == 0):
            return 0
        return float(tp)/(float(tp)+float(fn))
    elif (paras == 'Prevalence'):
        return float(tp)/(float(tp)+float(fp))
#     return tp,tn,fp,fn

# Change the model output to 0 or 1      
def binarilize(pred, standard):
    binary = []
    for i in range(len(pred)):
        if pred[i] > standard:
            binary.append(1)
        else:
            binary.append(0)
    return binary

# Compute the standard for a certain recall
def compute_standard(fixed_recall, y, pred):
    standard = 0
    standard_recall = 1
    for num in pred:
        tmp_pred = binarilize(pred, num)
        recall = conditions(y, tmp_pred, 'Recall')
        if (recall >= fixed_recall) and (recall < standard_recall):
            standard_recall = recall
            standard = num
    if (standard == 0):
        print('No matching valuse, please set a lower recall standard')
        return 0
    tmp = binarilize(pred, standard)
    PPV = conditions(y, tmp, 'Prevalence')
    return PPV