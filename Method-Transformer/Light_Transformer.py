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
MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### MOVE

df_eng = pd.read_pickle('../Data/eng_total.pkl')
del df_eng['id']

json_eng = df_eng.to_json('eng.json', orient='records', lines=True)

###

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
        #print(text, text_lengths)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)

def calculate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()

    return n_correct

### REMOVE
def categorical_accuracy(preds, y):
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
###

def accuracy(list_actual, list_prediction):
    actual = list_actual
    prediction = list_prediction

    precision = precision_score(actual, prediction, average=None, zero_division=0).tolist()
    recall = recall_score(actual, prediction, average=None, zero_division=0).tolist()

    f1_gen = f1_score(actual, prediction, average=None, zero_division=0).tolist()
    f1_micro = f1_score(actual, prediction, average='micro', zero_division=0).tolist()
    f1_macro = f1_score(actual, prediction, average='macro', zero_division=0).tolist()

    return precision, recall, f1_gen, f1_micro, f1_macro

def display(precisions, recalls, f1_gens, f1_micros, f1_macros):
    set1 = [precisions, recalls, f1_gens]
    set2 = [f1_micros, f1_macros]

    scores = []
    f1s = []

    for item in set1:
        tmp = np.array(item)
        scores_avg = np.multiply(np.mean(tmp, axis=0), 100).tolist()
        scores_std = np.multiply(np.std(tmp, axis=0), 100).tolist()

        scores_ite = []

        for i in range(len(scores_avg)):
            avg = "{:.2f}".format(scores_avg[i])
            std = "{:.2f}".format(scores_std[i])

            item_text = avg + "+-" + std
            scores_ite.append(item_text)

        scores.append(scores_ite)

    for item in set2:
        avg = "{:.2f}".format((statistics.mean(item)) * 100)
        std = "{:.2f}".format((statistics.stdev(item)) * 100)

        item_text = avg + "+-" + std
        f1s.append(item_text)

    df_score = pd.DataFrame(data=scores, index=['Precision', 'Recall', 'F1'], columns=CATEGORIES)
    df_average = pd.DataFrame(data=f1s, index=['F1 Microaverage', 'F1 Macroaverage'], columns=['Scores'])

    print(tabulate(df_score, headers='keys', tablefmt='pretty'))
    print(tabulate(df_average, headers='keys', tablefmt='pretty'))

def train(model, iterator, optimizer, loss_function):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0  
    model.train()
    
    for _, batch in tqdm(enumerate(iterator, 0)):
        outputs = model(batch.text)
        loss = loss_function(outputs, batch.label)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim = 1)
        n_correct += calculate_accuracy(big_idx, batch.label)

        nb_tr_steps += 1
        nb_tr_examples += batch.label.size(0)

        if _%5000==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct * 100)/nb_tr_examples

            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"The Total Accuracy for Epoch: {(n_correct * 100)/nb_tr_examples}")

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Test Training Loss Epoch: {epoch_loss}")
    print(f"Test Training Accuracy Epoch: {epoch_accu}")

def valid(model, iterator, loss_function):
    model.eval()

    n_correct = 0
    n_wrong = 0
    total = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0

    actual = []
    predicted = []
    
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
    
        for _, batch in tqdm(enumerate(iterator, 0)):
            outputs = model(batch.text)
            loss = loss_function(outputs, batch.label)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_accuracy(big_idx, batch.label)
            
            #acc = categorical_accuracy(outputs, batch.label)

            # epoch_loss += loss.item()
            # epoch_acc += acc.item()

            predicted.extend(big_idx.tolist())
            actual.extend(batch.label.tolist())

            nb_tr_steps += 1
            nb_tr_examples += batch.label.size(0)

            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct * 100)/nb_tr_examples

                print(f"Validation Loss per 100 Steps: {loss_step}")
                print(f"Validation Accuracy per 100 Steps: {accu_step}")

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples

    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    #return epoch_loss / len(iterator), epoch_acc / len(iterator)
    return accuracy(actual, predicted)

def test_CNN(json_file, lang):
    if lang == "EN":
        print("English")
    elif lang == "IS":
        print("Icelandic")

    precisions = []
    recalls = []
    f1_gens = []
    f1_micros = []
    f1_macros = []

    TEXT = data.Field(
        tokenize = 'spacy',
        tokenizer_language = 'en_core_web_sm',
        lower = True)
    LABEL = data.LabelField()

    fields = {'answer_freetext_value': ('text', TEXT), 'Sentiment': ('label', LABEL)}

    dataset = torchtext.legacy.data.TabularDataset(
            path=json_file,
            format="json",
            fields=fields)

    (train_data, test_data) = dataset.split(split_ratio=[0.8,0.2])

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

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters())

    loss_function = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    loss_function = loss_function.to(device)

    N_EPOCHS = 5

    for epoch in range(N_EPOCHS):
        train(model, train_iterator, optimizer, loss_function)
        #print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    precision, recall, f1_gen, f1_micro, f1_macro = valid(model, test_iterator, loss_function)
    
    print("F1: ", f1_gen)

def test_CNN_5fold(json_file, lang):
    if lang == "EN":
        print("English")
    elif lang == "IS":
        print("Icelandic")

    precisions = []
    recalls = []
    f1_gens = []
    f1_micros = []
    f1_macros = []

    TEXT = data.Field(
        tokenize = 'spacy',
        tokenizer_language = 'en_core_web_sm',
        lower = True)
    LABEL = data.LabelField()

    fields = {'answer_freetext_value': ('text', TEXT), 'Sentiment': ('label', LABEL)}

    dataset = torchtext.legacy.data.TabularDataset(
            path=json_file,
            format="json",
            fields=fields)

    for i in range(5):
        (train_data, test_data) = dataset.split(split_ratio=[0.8,0.2])

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

        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        optimizer = optim.Adam(model.parameters())

        loss_function = torch.nn.CrossEntropyLoss()

        model = model.to(device)
        loss_function = loss_function.to(device)

        N_EPOCHS = 5

        for epoch in range(N_EPOCHS):
            train(model, train_iterator, optimizer, loss_function)

        precision, recall, f1_gen, f1_micro, f1_macro = valid(model, test_iterator, loss_function)

        precisions.append(precision)
        recalls.append(recall)
        f1_gens.append(f1_gen)
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)

    display(precisions, recalls, f1_gens, f1_micros, f1_macros)

def test_biLSTM_5fold(json_file, lang):
    if lang == "EN":
        print("English")
    elif lang == "IS":
        print("Icelandic")

    precisions = []
    recalls = []
    f1_gens = []
    f1_micros = []
    f1_macros = []

    TEXT = data.Field(
        tokenize = 'spacy',
        tokenizer_language = 'en_core_web_sm',
        lower = True)
    LABEL = data.LabelField()

    fields = {'answer_freetext_value': ('text', TEXT), 'Sentiment': ('label', LABEL)}

    dataset = torchtext.legacy.data.TabularDataset(
            path=json_file,
            format="json",
            fields=fields)

    for i in range(5):
        (train_data, test_data) = dataset.split(split_ratio=[0.8,0.2])

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

        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        optimizer = optim.Adam(model.parameters())

        loss_function = torch.nn.BCEWithLogitsLoss()

        model = model.to(device)
        loss_function = loss_function.to(device)

        N_EPOCHS = 5

        for epoch in range(N_EPOCHS):
            train(model, train_iterator, optimizer, loss_function)

        precision, recall, f1_gen, f1_micro, f1_macro = valid(model, test_iterator, loss_function)

        precisions.append(precision)
        recalls.append(recall)
        f1_gens.append(f1_gen)
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)

    display(precisions, recalls, f1_gens, f1_micros, f1_macros)

test_biLSTM_5fold('eng.json', 'EN')