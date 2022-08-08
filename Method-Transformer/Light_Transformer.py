import json
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
from backup import BATCH_SIZE
import torchtext
from torchtext.legacy import data
from torchtext.legacy import datasets

import statistics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tabulate import tabulate
from tqdm import tqdm

CATEGORIES = ['Positive', 'Negative', 'Neutral']
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = torch.nn.ModuleList([
                                    torch.nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = torch.nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, text):
        text = text.permute(1, 0)
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)

        conved = [torch.nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))

        return self.fc(cat)

class biLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = torch.nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))

        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
            
        return self.fc(hidden)

def categorical_accuracy(preds, y):
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def CNN_train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text)
        
        loss = criterion(predictions, batch.label)
        
        acc = categorical_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def CNN_evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text)
            
            loss = criterion(predictions, batch.label)
            
            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def test_CNN(json_file, lang):
    if lang == 'EN':
        TEXT = data.Field(
            tokenize = 'spacy',
            tokenizer_language = 'en_core_web_sm',
            lower = True
        )
        
    elif lang == 'IS':
        TEXT = data.Field(
            lower = True
        )
    LABEL = data.LabelField()

    fields = {'answer_freetext_value': ('text', TEXT), 'Sentiment': ('label', LABEL)}

    dataset = torchtext.legacy.data.TabularDataset(
        path=json_file,
        format="json",
        fields=fields)

    (train_data, test_data) = dataset.split(split_ratio=[0.8,0.2])

    MAX_VOCAB_SIZE = 25_000

    TEXT.build_vocab(
            train_data, 
            max_size = MAX_VOCAB_SIZE, 
            vectors = 'glove.6B.100d',
            unk_init = torch.Tensor.normal_)

    LABEL.build_vocab(
            train_data)

    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data),
        device = device,
        batch_size = BATCH_SIZE,
        sort_key = lambda x: len(x.text),
        sort_within_batch = True)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [1,2,3]
    OUTPUT_DIM = len(LABEL.vocab)
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

    pretrained_embeddings = TEXT.vocab.vectors

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters())

    criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 5

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = CNN_train(model, train_iterator, optimizer, criterion)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    test_loss, test_acc = CNN_evaluate(model, test_iterator, criterion)
    print(test_acc)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        text, text_lengths = batch.text
        
        predictions = model(text, text_lengths).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, text_lengths = batch.text
            
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def test_biLSTM(json_file, lang):
    TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  include_lengths = True)

    LABEL = data.LabelField(dtype = torch.float)

    fields = {'answer_freetext_value': ('text', TEXT), 'Sentiment': ('label', LABEL)}

    dataset = torchtext.legacy.data.TabularDataset(
        path=json_file,
        format="json",
        fields=fields)

    (train_data, test_data) = dataset.split(split_ratio=[0.8,0.2])

    MAX_VOCAB_SIZE = 25_000

    TEXT.build_vocab(
            train_data, 
            max_size = MAX_VOCAB_SIZE, 
            vectors = 'glove.6B.100d',
            unk_init = torch.Tensor.normal_)

    LABEL.build_vocab(
            train_data)

    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data),
        device = device,
        batch_size = BATCH_SIZE,
        sort_key = lambda x: len(x.text),
        sort_within_batch = True)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = biLSTM(INPUT_DIM, 
                EMBEDDING_DIM, 
                HIDDEN_DIM, 
                OUTPUT_DIM, 
                N_LAYERS, 
                BIDIRECTIONAL, 
                DROPOUT, 
                PAD_IDX)

    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 5

    for epoch in range(N_EPOCHS):       
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(test_acc)