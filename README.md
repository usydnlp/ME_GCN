
# ME-GCN: Multi-dimensional Edge-Embedded Graph Convolutional Networks for Semi-supervised Text Classification
Official Implementation for ICLR 2022 on DLG4NLP
(https://arxiv.org/abs/2204.04618)

<h3>
  <b>Wang, K.*, Han, C.*, Long, S., & Poon, J. (2022). <br/><a href="https://arxiv.org/abs/2204.04618">ME-GCN: Multi-dimensional Edge-Embedded Graph Convolutional Networks for Semi-supervised Text Classification</a><br/>In proceeding of ICLR 2022 on DLG4NLP</b></span>
</h3>
*co first author


## Easy running using .ipynb file
You can simply run the code with your data using `final.ipynb`, remember to fill in your dataset into a list of documents/labels
```python
# ALL_INPUT =  a list of input sentences
# ALL_OUTPUT =  a list of output labels

# example 
# original_train_sentences = ['this is sample 1','this is sample 2']
# original_labels_train = ['postive','negative']

```
Also, some other parameters can be modified
```python

WORD_EMBEDDING = 0     # 0=word2vec, 1=fasttext, 2=glove, this is for word node embedding
DIM = 25               # number of streams
D2D_THRESHOLD = 15     # two documents sharing more than 15 words will have edges between them
POOLING = "max"        # "max","min","avg", the final pooling method

ALL_USED = False       # If True, only use partial of the dataset
USED_SIZE = 3000       # The number of samples used
TRAIN_PORTION = 0.01   # The proportion of labelled data

HIDDEN_DIM = 25
DROP_OUT = 0.5
LR = 0.002
WEIGHT_DECAY =  0
EPOCH = 2000
EARLY_STOPPING = 100

VAL_PORTION = 0.1
REMOVE_LESS_FREQUENT = 5
NUM_TEST = 5
```
