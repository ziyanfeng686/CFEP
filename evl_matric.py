
import math
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import time
import datetime
from collections import Counter
import copy
from collections import defaultdict

import random
random.seed(1234)

from scipy import interp
import warnings
warnings.filterwarnings("ignore")


from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data






def performances(y_true, y_pred, y_prob = None, rank_or_ic50 = True):
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel().tolist()
    print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    mcc = ((tp*tn) - (fn*fp)) / np.sqrt(np.float((tp+fn)*(tn+fp)*(tp+fp)*(tn+fn)))
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    
    try:
        recall = tp / (tp+fn)
    except:
        recall = 0
        
    try:
        precision = tp / (tp+fp)
    except:
        precision = 0
        
    try: 
        f1 = 2*precision*recall / (precision+recall)
    except:
        f1 = 0
    
    if y_prob is None:
        roc_auc, aupr = np.nan, np.nan
    else:
        if rank_or_ic50:
            roc_auc = roc_auc_score(y_true, 1 - y_prob)
            prec, reca, _ = precision_recall_curve(y_true,1 - y_prob)
        else:
            roc_auc = roc_auc_score(y_true, y_prob)
            prec, reca, _ = precision_recall_curve(y_true, y_prob)
            
        aupr = auc(reca, prec)

    print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
    print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
    print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
    print('auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}'.format(roc_auc, sensitivity, specificity, accuracy, mcc))
    print('precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}'.format(precision, recall, f1, aupr))

    return roc_auc, accuracy, mcc, f1, sensitivity, specificity, precision, recall, aupr

def compute_metric(pos_data,predictions,k=3):
    '''
    Compute Precision, Recall, and NDCG
    @pos_data: dict, key is [tcr]->[e1,e2,...]
    @predictions: dict, key is [tcr] -> [(e1,scores1),(e2,scores2), .... ] 
    '''
    recalls,precisions = [],[]
    ndcgs = []
    for hla in predictions.keys():
        try:
            pres, trues = predictions[hla], set(pos_data[hla])
        except KeyError:
            continue
        #print(trues)
        #print(pres)
        pres.sort(key=lambda x: -x[1])
        trues = [(p[0]) for p in trues]
        pres = [(p[0]) for p in pres]
        #print(trues)
        #print(pres)
        count = 0
        dcp = 0
        for i,p in enumerate(pres[:k]):
            if p in trues:
                count += 1
                dcp += 1 / np.log2(i+1 + 1)
        idcp = sum([1/np.log2(i+1+1) for i in range(min(len(trues),k))])
        ndcg = dcp / idcp if idcp > 0 else 0
        ndcgs.append(ndcg)
        precisions.append(count / k)
        recalls.append(count / len(trues))
    print('Precision, Recall, NDCG at top {} are:'.format(k))
    print(str(np.mean(precisions)) + ', ',str(np.mean(recalls)) + ', '+str(np.mean(ndcgs)))
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)


def benchmark_label_results(merged_data,pos_data):
    columns = merged_data.columns
    if 'prediction' in columns: 
        print('-----Performances according to Score threshold-----')
        merged_data['y_pred_score'] = [[0, 1][score >= 0.5] for score in merged_data.prediction]
        performance = performances(merged_data.Label, merged_data.y_pred_score, merged_data.prediction, rank_or_ic50 = False)
        pos_dict = pos_data.groupby('HLA_type').apply(lambda x: list(zip(x['MT_pep']))).to_dict()
        predictions_dict = data.groupby('HLA_type').apply(lambda x: list(zip(x['MT_pep'], x['prediction']))).to_dict()
        compute_metric(pos_dict,predictions_dict,k=100)
    return merged_data, performance


data = pd.read_csv(r'.\output\output.csv', index_col = 0)
pos_data = data[data['Label']==1]
benchmark_label_results(data,pos_data)














