import json
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
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

df_eng = pd.read_pickle('../Data/eng_total.pkl')
del df_eng['id']

json_eng = df_eng.to_json('eng.json', orient='records', lines=True)
# df_train, df_test = train_test_split(df_eng, test_size=0.2)
# json_train = df_train.to_json('train.json', orient='records', lines=True)
# json_test = df_test.to_json('test.json', orient='records', lines=True)

TEXT = data.Field(
        tokenize = 'spacy',
        tokenizer_language = 'en_core_web_sm',
        lower = True)
LABEL = data.LabelField()

fields = {'answer_freetext_value': ('text', TEXT), 'Sentiment': ('label', LABEL)}

# train_data, test_data = data.TabularDataset.splits(
#         path = '.',
#         train = 'train.json',
#         test = 'test.json',
#         format = 'json',
#         fields = fields)

dataset = torchtext.legacy.data.TabularDataset(
        path="eng.json",
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

# def accuracy(list_actual, list_prediction):
#     actual = list_actual
#     prediction = list_prediction

#     precision = precision_score(actual, prediction, average=None, zero_division=0).tolist()
#     recall = recall_score(actual, prediction, average=None, zero_division=0).tolist()

#     f1_gen = f1_score(actual, prediction, average=None, zero_division=0).tolist()
#     f1_micro = f1_score(actual, prediction, average='micro', zero_division=0).tolist()
#     f1_macro = f1_score(actual, prediction, average='macro', zero_division=0).tolist()

#     return precision, recall, f1_gen, f1_micro, f1_macro

# def display(precisions, recalls, f1_gens, f1_micros, f1_macros):
#     set1 = [precisions, recalls, f1_gens]
#     set2 = [f1_micros, f1_macros]

#     scores = []
#     f1s = []

#     for item in set1:
#         tmp = np.array(item)
#         scores_avg = np.multiply(np.mean(tmp, axis=0), 100).tolist()
#         scores_std = np.multiply(np.std(tmp, axis=0), 100).tolist()

#         scores_ite = []

#         for i in range(len(scores_avg)):
#             avg = "{:.2f}".format(scores_avg[i])
#             std = "{:.2f}".format(scores_std[i])

#             item_text = avg + "+-" + std
#             scores_ite.append(item_text)

#         scores.append(scores_ite)

#     for item in set2:
#         avg = "{:.2f}".format((statistics.mean(item)) * 100)
#         std = "{:.2f}".format((statistics.stdev(item)) * 100)

#         item_text = avg + "+-" + std
#         f1s.append(item_text)

#     df_score = pd.DataFrame(data=scores, index=['Precision', 'Recall', 'F1'], columns=CATEGORIES)
#     df_average = pd.DataFrame(data=f1s, index=['F1 Microaverage', 'F1 Macroaverage'], column=['Scores'])

#     print(tabulate(df_score, headers='keys', tablefmt='pretty'))
#     print(tabulate(df_average, headers='keys', tablefmt='pretty'))

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    device = device,
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch = True)

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

        self.encoder = torch.nn.LSTM(embedding_dim, 
                               hidden_dim, 
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               dropout=dropout)

        self.predictor = torch.nn.Linear(hidden_dim*2, output_dim)

        self.dropout = torch.nn.Dropout(dropout)
      
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))    
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        packed_output, (hidden, cell) = self.encoder(packed_embedded)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        return self.predictor(hidden)

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

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion):
    
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

def evaluate(model, iterator, criterion):
    
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

N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    #valid_loss, valid_acc = evaluate(model, test_iterator, criterion)
    
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(test_acc)
