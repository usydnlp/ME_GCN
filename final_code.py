# -*- coding: utf-8 -*-

"""
# Hyper Parameters
"""

WORD_EMBEDDING = 0     # 0=word2vec, 1=fasttext, 2=glove
DIM = 25               # number of streams
D2D_THRESHOLD = 15
POOLING = "max"        # "max","min","avg"

ALL_USED = False
USED_SIZE = 3000
TRAIN_PORTION = 0.01

HIDDEN_DIM = 25
DROP_OUT = 0.5
LR = 0.002
WEIGHT_DECAY =  0
EPOCH = 2000
EARLY_STOPPING = 100

VAL_PORTION = 0.1
REMOVE_LESS_FREQUENT = 5
NUM_TEST = 5

"""# Libraries"""

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import time

import math
from math import log
import scipy.sparse as sp

import nltk
from nltk.corpus import stopwords

from gensim.models import Word2Vec
from gensim.models import FastText
from glove import Corpus, Glove
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""# Download Dataset

"""

# ALL_OUTPUT =   a list of input sentences
# ALL_INPUT =  a list of output labels

"""# Preprocess

## Train Test Split
"""

if ALL_USED:
    sent_used, label_used = ALL_INPUT, ALL_OUTPUT
else:
    sent_used,_, label_used, _ = train_test_split(ALL_INPUT ,ALL_OUTPUT, train_size = USED_SIZE, stratify = ALL_OUTPUT, random_state = 0 )

not_all = False
try:
    train_sent, test_sent, train_labels, test_labels = train_test_split(sent_used ,label_used, stratify = label_used, train_size = TRAIN_PORTION, random_state = 0 ) 
except:
    train_sent, test_sent, train_labels, test_labels = train_test_split(sent_used ,label_used, train_size = TRAIN_PORTION, random_state = 0 ) 

unique_train = np.unique(train_labels)
unique_test = np.unique(test_labels)
for label in unique_test:
    if label not in unique_train:
        not_all = True        
        break

if not_all:
    labels_to_add = [label for label in unique_test if label not in unique_train]
    label_add_set = set(labels_to_add)
    i = 0
    while len(label_add_set)>0:
        label = test_labels[i]
        if label in label_add_set:
            train_sent.append(test_sent[i])
            train_labels.append(test_labels[i])
            test_sent = test_sent[:i]+test_sent[i+1:]
            test_labels = test_labels[:i]+test_labels[i+1:]
            label_add_set.remove(label)
        else:
            i += 1

original_sentences = train_sent+test_sent
train_size = len(train_sent)
test_size = len(test_sent)

"""## Label Encoding"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unique_labels=np.unique(train_labels + test_labels)

num_class = len(unique_labels)
lEnc = LabelEncoder()
lEnc.fit(unique_labels)

print(unique_labels)
print(lEnc.transform(unique_labels))

train_labels = lEnc.transform(train_labels)
test_labels = lEnc.transform(test_labels)
labels = train_labels.tolist()+test_labels.tolist()
labels = torch.LongTensor(labels).to(device)

"""## Remove Stopwords and less frequent words, tokenize sentences"""

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

original_word_freq = {}  # to remove rare words
for sentence in original_sentences:
    temp = clean_str(sentence)
    word_list = temp.split()
    for word in word_list:
        if word in original_word_freq:
            original_word_freq[word] += 1
        else:
            original_word_freq[word] = 1   

tokenize_sentences = []
word_list_dict = {}
for sentence in original_sentences:
    temp = clean_str(sentence)
    word_list_temp = temp.split()
    doc_words = []
    for word in word_list_temp:
        if word not in stop_words and original_word_freq[word] >= REMOVE_LESS_FREQUENT:
            doc_words.append(word)
            word_list_dict[word] = 1
    tokenize_sentences.append(doc_words)
word_list = list(word_list_dict.keys())
vocab_length = len(word_list)

del original_sentences

#word to id dict
word_id_map = {}
for i in range(vocab_length):
    word_id_map[word_list[i]] = i

"""## W2V"""

if WORD_EMBEDDING == 0:
    wv_cbow_model = Word2Vec(sentences=tokenize_sentences, size=DIM, window=5, min_count=0, workers=4, sg=0, iter=200)
    word_emb_dict = {}
    for word in word_list:
        word_emb_dict[word] = wv_cbow_model[word].tolist()
elif WORD_EMBEDDING == 1:
    ft_sg_model = FastText(sentences=tokenize_sentences, size=DIM, window=5, min_count=0, workers=4, sg=0, iter = 200)
    word_emb_dict = {}
    for word in word_list:
        word_emb_dict[word] = ft_sg_model[word].tolist()
elif WORD_EMBEDDING == 2:

    corpus = Corpus() 
    corpus.fit(tokenize_sentences, window=10)

    glove = Glove(no_components=DIM, learning_rate=0.05) 
    glove.fit(corpus.matrix, epochs=200, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)

    word_emb_dict = {}
    for word in word_list:
        word_emb_dict[word] = glove.word_vectors[glove.dictionary[word]].tolist()

"""## Doc2vec"""

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenize_sentences)]
model = Doc2Vec(documents, vector_size=DIM, window=5, min_count=1, workers=4, iter=200)

doc2vec_emb = []
for i in range(len(documents)):
    doc2vec_emb.append(model.docvecs[i])
doc2vec_npy = np.array(doc2vec_emb)

"""# Graph"""

node_size = train_size + vocab_length + test_size
adj_tensor = []

"""## d2w: tfidf"""

tfidf_row = []
tfidf_col = []
tfidf_weight = []

#get each word appears in which document
word_doc_list = {}
for word in word_list:
    word_doc_list[word]=[]

for i in range(len(tokenize_sentences)):
    doc_words = tokenize_sentences[i]
    unique_words = set(doc_words)
    for word in unique_words:
        exsit_list = word_doc_list[word]
        exsit_list.append(i)
        word_doc_list[word] = exsit_list

#document frequency
word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

# term frequency
doc_word_freq = {}

for doc_id in range(len(tokenize_sentences)):
    words = tokenize_sentences[doc_id]
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

for i in range(len(tokenize_sentences)):
    words = tokenize_sentences[i]
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row_tmp = i
        else:
            row_tmp = i + vocab_length
        col_tmp = train_size + j
        
        idf = log(1.0 * len(tokenize_sentences) / word_doc_freq[word_list[j]])
        weight_tmp = freq * idf
        doc_word_set.add(word)

        tfidf_row.append(row_tmp)
        tfidf_col.append(col_tmp)
        tfidf_weight.append(weight_tmp)

        tfidf_row.append(col_tmp)
        tfidf_col.append(row_tmp)
        tfidf_weight.append(weight_tmp)

"""## Diagonal"""

for i in range(node_size):
    tfidf_row.append(i)
    tfidf_col.append(i)
    tfidf_weight.append(1)

"""## w2w and d2d"""

def ordered_word_pair(a, b):
  if a > b:
    return (b, a)
  else:
    return (a, b)

co_dict = {}
for sent in tokenize_sentences:
    for i,word1 in enumerate(sent):
        for word2 in sent[i:]:
            co_dict[ordered_word_pair(word_id_map[word1],word_id_map[word2])] = 1

co_occur_threshold = D2D_THRESHOLD

doc_vec_bow = []
for sent in tokenize_sentences:
    temp = np.zeros((vocab_length))
    for word in sent:
        temp[word_id_map[word]] = 1
    doc_vec_bow.append(temp)

co_doc_dict = {}
for i in range(len(doc_vec_bow)-1):
    for j in range(i+1,len(doc_vec_bow)):
        if np.dot(doc_vec_bow[i],doc_vec_bow[j]) >= co_occur_threshold:
            co_doc_dict[(i,j)] = 1

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)

adj_list = []

for i in tqdm(range(DIM)):
    col = tfidf_col[:]
    row = tfidf_row[:]
    weight = tfidf_weight[:]
    for pair in co_dict:
        ind1, ind2 = pair

        word1 = word_list[ind1]
        word2 = word_list[ind2]
        tmp = np.tanh(1/np.abs(word_emb_dict[word1][i] - word_emb_dict[word2][i]))

        row.append(ind2+train_size)
        col.append(ind1+train_size)
        weight.append(tmp)

        row.append(ind1+train_size)
        col.append(ind2+train_size)
        weight.append(tmp)

    for pair in co_doc_dict:
        ind1, ind2 = pair        
        tmp = np.tanh(1/np.abs(doc2vec_npy[ind1][i] - doc2vec_npy[ind2][i]))

        if ind1>train_size:
            ind1 += vocab_length
        if ind2>train_size:    
            ind2 += vocab_length

        row.append(ind2)
        col.append(ind1)
        weight.append(tmp)

        row.append(ind1)
        col.append(ind2)
        weight.append(tmp)    

    
    adj_tmp = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))
    adj_tmp = adj_tmp + adj_tmp.T.multiply(adj_tmp.T > adj_tmp) - adj_tmp.multiply(adj_tmp.T > adj_tmp)
    adj_tmp = normalize_adj(adj_tmp) 
    adj_tmp = sparse_mx_to_torch_sparse_tensor(adj_tmp)
    adj_list.append(adj_tmp)

"""# Model - MULTIGCN

## input features - glove and doc2vec
"""

features = []
for i in range(train_size):
    features.append(doc2vec_npy[i])

for word in word_list:
    features.append(word_emb_dict[word])

for i in range(test_size):
    features.append(doc2vec_npy[train_size+i])

features = torch.FloatTensor(np.array(features)).to(device)

"""## GCN layer"""

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features,  drop_out = 0, activation=None, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.zeros(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(in_features, out_features)
        self.dropout = torch.nn.Dropout(drop_out)
        self.activation =  activation

    def reset_parameters(self,in_features, out_features):
        stdv = np.sqrt(6.0/(in_features+out_features))
        self.weight.data.uniform_(-stdv, stdv)


    def forward(self, input, adj, feature_less=False):
        if feature_less:
            support = self.weight
        else:
            input = self.dropout(input)
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

"""## Main Model"""

class MULTIGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MULTIGCN, self).__init__()

        # different weights
        self.intras1 = nn.ModuleList([GraphConvolution(nfeat, nhid, dropout, activation = nn.ReLU()) for i in range(DIM)])
        self.intras2 = nn.ModuleList([GraphConvolution(nhid*DIM, nclass, dropout, activation = nn.ReLU()) for i in range(DIM)])


    def forward(self, x, adj, feature_less=False):
        x = torch.stack([self.intras1[i](x,adj[i],feature_less) for i in range(DIM)]) 
        x = x.permute(1,0,2) 
        x = x.reshape(x.size()[0],-1)  
        x = torch.stack([self.intras2[i](x,adj[i]) for i in range(DIM)]) 

 
        if POOLING == 'avg':
            return torch.mean(x,0)
        if POOLING == 'max':
            return torch.max(x,0)[0]
        if POOLING == 'min':
            return torch.min(x,0)[0]

"""## Training"""

real_train_size = int((1-VAL_PORTION)*train_size)
val_size = train_size-real_train_size

idx_train = range(real_train_size)
idx_val = range(real_train_size,train_size)
idx_test = range(train_size + vocab_length,node_size)

# Model and optimizer

def cal_accuracy(predictions,labels):
    pred = torch.argmax(predictions,-1).cpu().tolist()
    lab = labels.cpu().tolist()
    cor = 0
    for i in range(len(pred)):
        if pred[i] == lab[i]:
            cor += 1
    return cor/len(pred)


final_acc_list = []
for _ in range(NUM_TEST):
    model = MULTIGCN(nfeat=features.shape[1], nhid=HIDDEN_DIM, nclass=num_class, dropout=DROP_OUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    val_loss = []
    for epoch in range(EPOCH):

        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_list)
        loss_train = criterion(output[idx_train], labels[idx_train])
        acc_train = cal_accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()


        model.eval()
        output = model(features, adj_list)

        loss_val = criterion(output[idx_val], labels[idx_val])
        val_loss.append(loss_val.item())
        acc_val = cal_accuracy(output[idx_val], labels[idx_val])
        print(  'Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'acc_train: {:.4f}'.format(acc_train),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val),
                'time: {:.4f}s'.format(time.time() - t))
        
        if epoch > EARLY_STOPPING and np.min(val_loss[-EARLY_STOPPING:]) > np.min(val_loss[:-EARLY_STOPPING]) :
            print("Early Stopping...")
            break

    model.eval()
    output = model(features, adj_list)
    loss_test = criterion(output[idx_test], labels[-test_size:])
    acc_test = cal_accuracy(output[idx_test], labels[-test_size:])
    print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test))

    final_acc_list.append(acc_test)

    print(classification_report(test_labels,torch.argmax(output[idx_test],-1).cpu().tolist(),digits = 4))

print(np.round(final_acc_list,4))