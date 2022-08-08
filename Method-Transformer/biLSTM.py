import pandas as pd
import numpy as np
import time
import spacy
import random
from pathlib import Path
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.legacy import data 
import torchtext
from nltk.tokenize import wordpunct_tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LEARNING_RATE = 1e-05
CATEGORIES = ['Positive', 'Negative', 'Neutral']

encoding_dict = {
                'positive':0,
                'neutral':2,
                'negative':1,
                'negativa':1
                }

torch.backends.cudnn.deterministic = True

df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", header=None)

df[0]=df[0].replace(to_replace=4,value=1)

df.sample(50000).to_csv("sentiment140-small.csv", encoding='utf-8', header=None, index=None)

TEXT = data.Field(tokenize='spacy', lower=True, include_lengths= True)

LABEL = data.LabelField(dtype=torch.float)

fields = [('label', LABEL), ('id',None),('date',None),('query',None),
      ('name',None), ('text', TEXT), ('category',None)]

dataset = torchtext.legacy.data.TabularDataset(
        path="sentiment140-small.csv",
        format="CSV",
        fields=fields,
        skip_header=False)

(train_data, test_data, valid_data) = dataset.split(split_ratio=[0.8,0.1,0.1])

MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE,
                 vectors = "glove.6B.100d",
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    device = device,
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch = True)

class biLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        self.encoder = nn.LSTM(embedding_dim, 
                               hidden_dim, 
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               dropout=dropout)

        self.predictor = nn.Linear(hidden_dim*2, output_dim)

        self.dropout = nn.Dropout(dropout)
      
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))    
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        packed_output, (hidden, cell) = self.encoder(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        return self.predictor(hidden)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

# Create an instance of LSTM class
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

optimizer = optim.Adam(model.parameters(), lr=2e-2)

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

def batch_accuracy(predictions, label):
    preds = torch.round(torch.sigmoid(predictions))
    correct = (preds == label).float()
    accuracy = correct.sum() / len(correct)

    return accuracy

def train(model, iterator, optimizer, criterion):    
    training_loss = 0.0
    training_acc = 0.0
    
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        accuracy = batch_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()
        
        training_loss += loss.item()
        training_acc += accuracy.item()

    return training_loss / len(iterator), training_acc / len(iterator)

def evaluate(model, iterator, criterion):
    eval_loss = 0.0
    eval_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, text_lengths = batch.text
            
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            accuracy = batch_accuracy(predictions, batch.label)

            eval_loss += loss.item()
            eval_acc += accuracy.item()
        
    return eval_loss / len(iterator), eval_acc / len(iterator)

NUM_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model-small.pt')

    print("Epoch {}:".format(epoch+1))
    print("\t Train Loss {} | Train Accuracy: {}%".format(round(train_loss, 2), round(train_acc*100, 2)))
    print("\t Validation Loss {} | Validation Accuracy: {}%".format(round(valid_loss, 2), round(valid_acc*100, 2)))

# Load the model with the best validation loss
model.load_state_dict(torch.load('model-small.pt'))

# Evaluate test loss and accuracy
test_loss, test_acc = evaluate(model, test_iterator, criterion)

nlp = spacy.load('en_core_web_sm')

def predict(model, text, tokenized=True):
    model.eval()

    if tokenized == False:
        tokens = [token.text for token in nlp.tokenizer(text)]
    else:
        tokens = text

    indexed_tokens = [TEXT.vocab.stoi[t] for t in tokens]
    length = [len(indexed_tokens)]
    tensor = torch.LongTensor(indexed_tokens).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))

    return prediction.item()

# List to append data to
# d = []


# for idx in range(10):

#     # Detokenize the tweets from the test set
#     tweet = TreebankWordDetokenizer().detokenize(test_data[idx].text)
                                                 
#     # Append tweet, prediction, and true label
#     d.append({'Tweet': tweet, 'Prediction': predict(model, test_data[idx].text), 'True Label': test_data[idx].label})

# # Convert list to dataframe
# pd.DataFrame(d)
