
from __future__ import division
import math
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import time
import datetime
import random
random.seed(1234)
import pickle

import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy
from sklearn.model_selection import KFold
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
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'





seed = 20220828
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True




hla_sequence = pd.read_csv(r'./data/common_hla_sequence.csv')





def make_data(data):
    pep_inputs, hla_inputs, labels = [], [], []
    for pep, hla, label in zip(data.MT_pep, data.HLA_sequence, data.Label):
        pep, hla = pep.ljust(pep_max_len, '-'), hla.ljust(hla_max_len, '-')
        pep_input = [[vocab[n] for n in pep]] # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        hla_input = [[vocab[n] for n in hla]]
        pep_inputs.extend(pep_input)
        hla_inputs.extend(hla_input)
        labels.append(label)
    return torch.LongTensor(pep_inputs), torch.LongTensor(hla_inputs), torch.LongTensor(labels)

class MyDataSet(Data.Dataset):
    def __init__(self, pep_inputs, hla_inputs, labels):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.hla_inputs = hla_inputs
        self.labels = labels

    def __len__(self): 
        return self.pep_inputs.shape[0] 

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.hla_inputs[idx], self.labels[idx]





class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        #self.use_cuda = use_cuda
        #device = torch.device("cuda" if self.use_cuda else "cpu")
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        #return nn.LayerNorm(d_model).to(device)(output + residual), attn

        return output+residual,attn

class ConvModule(nn.Module):
    def __init__(self, input_dim, kernel_size, dropout_rate, causal=False):
        super(ConvModule, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)

        self.pw_conv_1 = nn.Conv2d(1, 2, 1, 1, 0)
        self.glu_act = torch.nn.Sigmoid()
        self.causal = causal
        if causal:
            self.dw_conv_1d = nn.Conv1d(input_dim, input_dim, kernel_size, 1, padding=(kernel_size-1), groups=input_dim)
        else:
            self.dw_conv_1d = nn.Conv1d(input_dim, input_dim, kernel_size, 1, padding=(kernel_size-1)//2, groups=input_dim)
        self.BN = nn.BatchNorm1d(input_dim)
        self.act = nn.ReLU()
        self.pw_conv_2 = nn.Conv2d(1, 1, 1, 1, 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.kernel_size = kernel_size

    def forward(self, inputs):
        residual = inputs
        x = inputs.unsqueeze(1)
        x = self.layer_norm(x)
        x = self.pw_conv_1(x)
        x = x[:, 0] * self.glu_act(x[:, 1])
        x = x.permute([0, 2, 1])
        x = self.dw_conv_1d(x)
        if self.causal:
            x = x[:, :, :-(self.kernel_size-1)]
        x = self.BN(x)
        x = self.act(x)
        x = x.unsqueeze(1).permute([0, 1, 3, 2])
        x = self.pw_conv_2(x)
        x = self.dropout(x).squeeze(1)
        return x+residual

   
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        #self.use_cuda = use_cuda
        #device = torch.device("cuda" if self.use_cuda else "cpu")
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual) # [batch_size, seq_len, d_model]

    
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.conv = ConvModule(d_model, kernel_size, dropout_rate, causal)
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.conv(enc_outputs)
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns





class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.conv = ConvModule(d_model, kernel_size, dropout_rate, causal)
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, dec_self_attn_mask): # dec_inputs = enc_outputs
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.conv(dec_outputs)
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn

    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        #self.tgt_emb = nn.Embedding(d_model * 2, d_model)
        #self.use_cuda = use_cuda
        #device = torch.device("cuda" if self.use_cuda else "cpu")
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.tgt_len = tgt_len
        
    def forward(self, dec_inputs): # dec_inputs = enc_outputs (batch_size, peptide_hla_maxlen_sum, d_model)
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
#         dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1).to(device) # [batch_size, tgt_len, d_model]
#         dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]
        dec_self_attn_pad_mask = torch.LongTensor(np.zeros((dec_inputs.shape[0], tgt_len, tgt_len))).bool().to(device)

        dec_self_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)
            
        return dec_outputs, dec_self_attns

class Conv1DModel(nn.Module):
    def __init__(self):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=45, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
       
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.fc1 = nn.Linear(128 * 32, 256) 
        self.relu2 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256) 
        self.fc2 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU()  
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
       
        x = x.view(x.size(0), -1)  
        
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.fc3(x)
        
        
        return x






class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.use_cuda = use_cuda
        device = torch.device("cuda" if use_cuda else "cpu")
        self.pep_encoder = Encoder().to(device)
        self.hla_encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.tgt_len = tgt_len
        self.cnn_pre = Conv1DModel().to(device)
        
    def forward(self, pep_inputs, hla_inputs):
        '''
        pep_inputs: [batch_size, pep_len]
        hla_inputs: [batch_size, hla_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        pep_enc_outputs, pep_enc_self_attns = self.pep_encoder(pep_inputs)
        hla_enc_outputs, hla_enc_self_attns = self.hla_encoder(hla_inputs)
        enc_outputs = torch.cat((pep_enc_outputs, hla_enc_outputs), 1) # concat pep & hla embedding
        
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns = self.decoder(enc_outputs)
        
        
        #dec_outputs = dec_outputs.view(dec_outputs.shape[0], -1) # Flatten [batch_size, tgt_len * d_model]
        dec_logits = self.cnn_pre(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]

        return dec_logits.view(-1, dec_logits.size(-1)), pep_enc_self_attns, hla_enc_self_attns, dec_self_attns




def performances(y_true, y_pred, y_prob, print_ = True):
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel().tolist()
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    try:
        mcc = ((tp*tn) - (fn*fp)) / np.sqrt(np.float((tp+fn)*(tn+fp)*(tp+fp)*(tn+fn)))
    except:
        print('MCC Error: ', (tp+fn)*(tn+fp)*(tp+fp)*(tn+fn))
        mcc = np.nan
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    
    try:
        recall = tp / (tp+fn)
    except:
        recall = np.nan
        
    try:
        precision = tp / (tp+fp)
    except:
        precision = np.nan
        
    try: 
        f1 = 2*precision*recall / (precision+recall)
    except:
        f1 = np.nan
        
    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)
    
    if print_:
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
        print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
        print('auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}'.format(roc_auc, sensitivity, specificity, accuracy, mcc))
        print('precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}'.format(precision, recall, f1, aupr))
    
    return (roc_auc, precision, recall, aupr, accuracy, mcc, f1, sensitivity, specificity)

def transfer(y_prob, threshold = 0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])


def performances_to_pd(performances_dict):
    metrics_name = ['roc_auc', 'precision', 'recall', 'aupr','accuracy', 'mcc', 'f1', 'sensitivity', 'specificity', ]
    performances_pd = pd.DataFrame(performances_dict, columns = metrics_name)
    return performances_pd





f_mean = lambda l: sum(l)/len(l)


def train_step(model, train_loader, fold, epoch, epochs, use_cuda = True):
    device = torch.device("cuda" if use_cuda else "cpu")
    
    time_train_ep = 0
    model.train()
    y_true_train_list, y_prob_train_list = [], []
    loss_train_list, dec_attns_train_list = [], []
    for train_pep_inputs, train_hla_inputs, train_labels in tqdm(train_loader):
        '''
        pep_inputs: [batch_size, pep_len]
        hla_inputs: [batch_size, hla_len]
        train_outputs: [batch_size, 2]
        '''
        train_pep_inputs, train_hla_inputs, train_labels = train_pep_inputs.to(device), train_hla_inputs.to(device), train_labels.to(device)

        t1 = time.time()
        train_outputs, _, _, train_dec_self_attns = model(train_pep_inputs, 
                                                                                                        train_hla_inputs)
        train_loss = criterion(train_outputs, train_labels)
        time_train_ep += time.time() - t1

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        y_true_train = train_labels.cpu().numpy()
        y_prob_train = nn.Softmax(dim = 1)(train_outputs)[:, 1].cpu().detach().numpy()
        
        y_true_train_list.extend(y_true_train)
        y_prob_train_list.extend(y_prob_train)
        loss_train_list.append(train_loss)
#         dec_attns_train_list.append(train_dec_self_attns)
        
    y_pred_train_list = transfer(y_prob_train_list, threshold)
    ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)
    
    print('Fold-{}****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec'.format(fold, epoch, epochs, f_mean(loss_train_list), time_train_ep))
    metrics_train = performances(y_true_train_list, y_pred_train_list, y_prob_train_list, print_ = True)
    
    return ys_train, loss_train_list, metrics_train, time_train_ep#, dec_attns_train_list


def eval_step(model, val_loader, fold, epoch, epochs, use_cuda = True):
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model.eval()
    torch.manual_seed(19961231)
    torch.cuda.manual_seed(19961231)
    with torch.no_grad():
        loss_val_list, dec_attns_val_list = [], []
        y_true_val_list, y_prob_val_list = [], []
        for val_pep_inputs, val_hla_inputs, val_labels in tqdm(val_loader):
            val_pep_inputs, val_hla_inputs, val_labels = val_pep_inputs.to(device), val_hla_inputs.to(device), val_labels.to(device)
            val_outputs, _, _, val_dec_self_attns = model(val_pep_inputs, val_hla_inputs)
            val_loss = criterion(val_outputs, val_labels)

            y_true_val = val_labels.cpu().numpy()
            y_prob_val = nn.Softmax(dim = 1)(val_outputs)[:, 1].cpu().detach().numpy()

            y_true_val_list.extend(y_true_val)
            y_prob_val_list.extend(y_prob_val)
            loss_val_list.append(val_loss)
#             dec_attns_val_list.append(val_dec_self_attns)
            
        y_pred_val_list = transfer(y_prob_val_list, threshold)
        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)
        
        print('Fold-{} ****Test  Epoch-{}/{}: Loss = {:.6f}'.format(fold, epoch, epochs, f_mean(loss_val_list)))
        metrics_val = performances(y_true_val_list, y_pred_val_list, y_prob_val_list, print_ = True)
    return ys_val, loss_val_list, metrics_val#, dec_attns_val_list





pep_max_len = 11 # peptide; enc_input max sequence length
hla_max_len = 34 # hla; dec_input(=dec_output) max sequence length
tgt_len = pep_max_len + hla_max_len
pep_max_len, hla_max_len

vocab = np.load(r'./data/vocab_dict.npy', allow_pickle = True).item()
vocab_size = len(vocab)





# Transformer Parameters
d_model = 64  # Embedding Size
d_ff = 512 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 1  # number of Encoder of Decoder Layer
n_heads = 8
kernel_size = 3
dropout_rate = 0.1
causal=False
batch_size = 1024
epochs = 50
threshold = 0.5

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")





def data_with_loader(type_ = 'train',fold = None,  batch_size = 1024):
    if type_ != 'train' and type_ != 'val':
        data = pd.read_csv('./data/CFEPI/{}_data.csv'.format(type_), index_col = False)

    elif type_ == 'train':
        data = pd.read_csv('./data/CFEPI/train_data.csv', index_col = False)
    elif type_ == 'val':
        data = pd.read_csv('./data/CFEPI/val_data.csv', index_col = False)
        
    pep_inputs, hla_inputs, labels = make_data(data)
    loader = Data.DataLoader(MyDataSet(pep_inputs, hla_inputs, labels), batch_size, shuffle = False, num_workers = 0)
    
    return data, pep_inputs, hla_inputs, labels, loader



#independent_data, independent_pep_inputs, independent_hla_inputs, independent_labels, independent_loader = data_with_loader(type_ = 'tsest',fold = None,  batch_size = batch_size)
#external_data, external_pep_inputs, external_hla_inputs, external_labels, external_loader = data_with_loader(type_ = 'external',fold = None,  batch_size = batch_size)





ys_train_fold_dict, ys_val_fold_dict = {}, {}
train_fold_metrics_list, val_fold_metrics_list = [], []
independent_fold_metrics_list = []
external_fold_metrics_list = []
attns_train_fold_dict, attns_val_fold_dict= {}, {}
loss_train_fold_dict, loss_val_fold_dict = {}, {}




for fold in range(0,1):
    print('=====Fold-{}====='.format(fold))
    print('-----Generate data loader-----')

    train_data, train_pep_inputs, train_hla_inputs, train_labels, train_loader = data_with_loader(type_ = 'train', fold = fold,  batch_size = batch_size)
    val_data, val_pep_inputs, val_hla_inputs, val_labels, val_loader = data_with_loader(type_ = 'val', fold = fold,  batch_size = batch_size)
    print('Fold-{} Label info: Train = {} | Val = {}'.format(fold, Counter(train_data.Label), Counter(val_data.Label)))

    print('-----Compile model-----')
    model = Transformer().to(device)  
    weight = torch.tensor([3.0, 1.0],device = device)
    criterion = nn.CrossEntropyLoss(weight = weight)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)#, momentum = 0.99)

    print('-----Train-----')
    dir_saver = r'./checkpoints/CFEPI'
    path_saver = r'./checkpoints/CFEPI/fold{}.pkl'.format(fold)
    print('dir_saver: ', dir_saver)
    print('path_saver: ', path_saver)

    metric_best, ep_best = 0, 1
    time_train = 0
 
    for epoch in range(1, epochs + 1):

        ys_train, loss_train_list, metrics_train, time_train_ep = train_step(model, train_loader, fold, epoch, epochs, use_cuda) # , dec_attns_train
        ys_val, loss_val_list, metrics_val = eval_step(model, val_loader, fold, epoch, epochs, use_cuda) #, dec_attns_val

        metrics_ep_avg = (metrics_val[0] + metrics_val[3]) / 2
        if metrics_ep_avg > metric_best: 
            metric_best, ep_best = metrics_ep_avg, epoch
            if not os.path.exists(dir_saver):
                os.makedirs(dir_saver)
            print('****Saving model: Best epoch = {} | 5metrics_Best_avg = {:.4f}'.format(ep_best, metric_best))
            print('*****Path saver: ', path_saver)
            torch.save(model.eval().state_dict(), path_saver)

        time_train += time_train_ep

    print('-----Optimization Finished!-----')
    print('-----Evaluate Results-----')
    

    if ep_best >= 0:
        print('*****Path saver: ', path_saver)
        model.load_state_dict(torch.load(path_saver))
        model_eval = model.eval()

        ys_res_train, loss_res_train_list, metrics_res_train = eval_step(model_eval, train_loader, fold, ep_best, epochs, use_cuda) # , train_res_attns
        ys_res_val, loss_res_val_list, metrics_res_val = eval_step(model_eval, val_loader, fold, ep_best, epochs, use_cuda) # , val_res_attns
        #ys_res_independent, loss_res_independent_list, metrics_res_independent = eval_step(model_eval, independent_loader, fold, ep_best, epochs, use_cuda) # , independent_res_attns
        #ys_res_external, loss_res_external_list, metrics_res_external = eval_step(model_eval, external_loader, fold, ep_best, epochs, use_cuda) # , external_res_attns

        
        
        
        train_fold_metrics_list.append(metrics_res_train)
        val_fold_metrics_list.append(metrics_res_val)
        
        #independent_fold_metrics_list.append(metrics_res_independent)
        #external_fold_metrics_list.append(metrics_res_external)

        ys_train_fold_dict[fold], ys_val_fold_dict[fold] = ys_res_train, ys_res_val   
        loss_train_fold_dict[fold], loss_val_fold_dict[fold] = loss_res_train_list, loss_res_val_list

    print("Total training time: {:6.2f} sec".format(time_train))





print('****Train set:')
d3 = performances_to_pd(train_fold_metrics_list)
print(d3)
d3.to_csv('train.csv')

print('****Val set:')

d4 = performances_to_pd(val_fold_metrics_list)
print(d4)
d4.to_csv('val.csv')

#print('****Independent set:')
#d5 = performances_to_pd(independent_fold_metrics_list)
#print(d5)
#d5.to_csv('independent_text.csv')

#print('****External set:')
#d6 = performances_to_pd(external_fold_metrics_list)
#d6.to_csv('external_text.csv')