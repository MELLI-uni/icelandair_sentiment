# Reference:
#   https://blahblahlab.tistory.com/135
#   https://srinivas-yeeda.medium.com/sentiment-analysis-using-word2vec-and-glove-embeddings-5ad7d50ddb0d
#   https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/5%20-%20Multi-class%20Sentiment%20Analysis.ipynb
#   https://www.knime.com/blog/lexicon-based-sentiment-analysis
#   https://www.icelandicmadeeasier.com/posts/basic-word-order
#   http://learn101.org/icelandic_grammar.php
#   https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb
#   https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb
#   https://github.com/stofnun-arna-magnussonar/ordgreypingar_embeddings/blob/c3c6759eded5421e39ef14ce89135039b2bc8edc/word2vec/train_w2v.py#L65

import xlwings as xws
import regex as re
import string

import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statistics

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tabulate import tabulate
from tqdm import tqdm

import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from torchtext import data, datasets

device = 'cuda' if cuda.is_available() else 'cpu'

TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)

trainset , testset = datasets.IMDB.splits(TEXT,LABEL)

print(vars(trainset.examples[0]))

# print(testset)

# TEXT.build_vocab(trainset, min_freq=5)
# LABEL.build_vocab(trainset)

# vocab_size = len(TEXT.vocab)
# n_classes = 3

# class CNN(torch.nn.Module):
#     def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
#         super().__init__()

#         self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

#         self.convs = torch.nn.ModuleList([
#             torch.nn.Conv2d(in_channels = 1,
#                 out_channels = n_filters,
#                 kernel_size = (fs, embedding_dim))
#             for fs in filter_sizes
#         ])

#         self.fc = torch.nn.Linear(len(filter_sizes) * n_filters, output_dim)
#         self.dropout = torch.nn.Dropout(dropout)

#     def forward(self, text):
#         text = text.permute(1, 0)
#         embedded = self.embedding(text)
#         embedded = embedded.unsqueeze(1)
#         conved = [torch.nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
#         pooled = [torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
#         cat = self.dropout(torch.cat(pooled, dim=1))

#         return self.fc(cat)

class cLSTM(torch.nn.Module):
    def __init__(self, n_layers, hidden_size, n_vocab, embedding_size, n_classes, num_dirs=1, dropout=0.5):
        super(cLSTM.self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embed = torch.nn.Embedding(vocab_size, embedding_size)
        self.num_dirs = num_dirs

        self.lstm = torch.nn.LSTM(
            input_size = embedding_size,
            hidden_size = hidden_size,
            num_layers = n_layers,
            batch_first = True,
            bidirectional = True if num_dirs > 1 else False
        )

        self.out = torch.nn.Linear(hidden_size * n_layers, n_classes)
        self.droout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.embeded(x)
        h_0 = torch.zeros((self.n_layers * self.num_dirs, x.shape[0], self.hidden_size)).to(device)
        c_0 = torch.zeros((self.n_layers * self.num_dirs, x.shape[0], self.hidden_size)).to(device)

        hidden_states, h_n, c_n = self.lstm(x, (h_0, c_0))
        self.droptout(h_n)

        h_n = h_n.view(h_n.shape[1], -1)

        logit = self.out(h_n)

        return logit

# def calculate_accuracy(preds, y):
#     top_pred = preds.argmax(1, keepdim-True)
#     correct = top_pred.eq(y.view_as(top_pred)).sum()
#     acc = correct.float() / y.shape[0]

#     return acc

# def train(model, iterator, loss_function, optimizer):
#     epoch_loss = 0
#     epoch_acc = 0

#     model.train()

#     for batch in iterator:
#         optimizer.zero_grad()

#         predictions = model(batch.text)
#         loss = loss_function(predictions, batch.label)
#         acc = calculate_accuracy(predictions, batch.label)

#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()
#         epoch_acc = acc.item()

#     return epoch_loss / len(iterator), epoch_acc / len(iterator)

# def evaluate(model, iterator, loss_function):
#     epoch_loss = 0
#     epoch_acc = 0
    
#     model.eval()

#     with torch.no_grad():
#         for batch in iterator:
#             predictions = model(batch.text)

#             loss = loss_function(predictions, batch.label)
#             acc = calculate_accuracy(predictions, batch.label)

#             epoch_loss += loss.item()
#             epoch_acc += acc.item()

#     return epoch_loss / len(iterator), epoch_acc / len(iterator)

# def predict_class(model, sentence, min_len = 4):
#     model.eval()

#     if len(tokenized) < min_len:
#         tokenized += ['<pad>'] * (min_len - len(tokenized))

#     indexed = [TEXT.vocab.stoi[t] for t in tokenized]
#     tensor = torch.LongTensor(indexed).to(device)
#     tensor = tensor.unsqueeze(1)
#     preds = model(tensor)
#     max_preds = preds.argmax(dim = 1)

#     return max_preds.item()

# INPUT_DIM = len(TEXT.vocab)
# EMBEDDING_DIM = 200
# N_FILTERS = 100
# FILTER_SIZES = [3, 4, 5]
# OUTPUT_DIM = len(LABEL.vocab)
# DROPOUT = 0.5
# PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
# model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

# pretrained_embeddings = TEXT.vocab.vectors
# model.embedding.weight.data.copy_(pretrained_embeddings)

# UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

# model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
# model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

# optimizer = optim.Adam(model.parameters())
# criterion = nn.CrossEntropyLoss()

# model = model.to(device)
# criterion = criterion.to(device)

#pred_class = predict_class(model, "sentence")