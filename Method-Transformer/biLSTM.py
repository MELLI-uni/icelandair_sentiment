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
# Setting device on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
# print()

# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

torch.backends.cudnn.deterministic = True

# Read in data into a dataframe
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", header=None)

# df.head(5)

df[0]=df[0].replace(to_replace=4,value=1)

df.sample(50000).to_csv("sentiment140-small.csv", encoding='utf-8', header=None, index=None)

# Declare fields for tweets and labels
# include_lengths tells the RNN how long the actual sequences are
TEXT = data.Field(tokenize='spacy', lower=True, include_lengths= True)
LABEL = data.LabelField(dtype=torch.float)

# Map data to fields
fields = [('label', LABEL), ('id',None),('date',None),('query',None),
      ('name',None), ('text', TEXT),('category',None)]

# Apply field definition to create torch dataset
dataset = torchtext.legacy.data.TabularDataset(
        path="sentiment140-small.csv",
        format="CSV",
        fields=fields,
        skip_header=False)

# Split data into train, test, validation sets
(train_data, test_data, valid_data) = dataset.split(split_ratio=[0.8,0.1,0.1])

# print("Number of train data: {}".format(len(train_data)))
# print("Number of test data: {}".format(len(test_data)))
# print("Number of validation data: {}".format(len(valid_data)))

MAX_VOCAB_SIZE = 25000

# unk_init initializes words in the vocab using the Gaussian distribution
TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE,
                 vectors = "glove.6B.100d",
                 unk_init = torch.Tensor.normal_)

# build vocab for training set - convert words into integers
LABEL.build_vocab(train_data)

# Most frequent tokens
TEXT.vocab.freqs.most_common(10)

BATCH_SIZE = 128

# sort_within_batch sorts all the tensors within a batch by their lengths
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    device = device,
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch = True)

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        """
        Define the layers of the module.

        vocab_size - vocabulary size
        embedding_dim - size of the dense word vectors
        hidden_dim - size of the hidden states
        output_dim - number of classes
        n_layers - number of multi-layer RNN
        bidirectional - boolean - use both directions of LSTM
        dropout - dropout probability
        pad_idx -  string representing the pad token
        """
        
        super().__init__()

        # 1. Feed the tweets in the embedding layer
        # padding_idx set to not learn the emedding for the <pad> token - irrelevant to determining sentiment
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        # 2. LSTM layer
        # returns the output and a tuple of the final hidden state and final cell state
        self.encoder = nn.LSTM(embedding_dim, 
                               hidden_dim, 
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               dropout=dropout)
        
        # 3. Fully-connected layer
        # Final hidden state has both a forward and a backward component concatenated together
        # The size of the input to the nn.Linear layer is twice that of the hidden dimension size
        self.predictor = nn.Linear(hidden_dim*2, output_dim)

        # Initialize dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
      
    def forward(self, text, text_lengths):
        """
        The forward method is called when data is fed into the model.

        text - [tweet length, batch size]
        text_lengths - lengths of tweet
        """

        # embedded = [sentence len, batch size, emb dim]
        embedded = self.dropout(self.embedding(text))    

        # Pack the embeddings - cause RNN to only process non-padded elements
        # Speeds up computation
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))

        # output of encoder
        packed_output, (hidden, cell) = self.encoder(packed_embedded)

        # unpack sequence - transform packed sequence to a tensor
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sentence len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors
        
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        
        # Get the final layer forward and backward hidden states  
        # concat the final forward and backward hidden layers and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        # hidden = [batch size, hid dim * num directions]

        return self.predictor(hidden)

INPUT_DIM = len(TEXT.vocab)
# dim must be equal to the dim of pre-trained GloVe vectors
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
# 2 layers of biLSTM
N_LAYERS = 2
BIDIRECTIONAL = True
# Dropout probability
DROPOUT = 0.5
# Get pad token index from vocab
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

# Create an instance of LSTM class
model = LSTM(INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
            PAD_IDX)

pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)

# Initialize <unk> and <pad> both to all zeros - irrelevant for sentiment analysis
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

# Setting row in the embedding weights matrix to zero using the token index
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

# Adam optimizer used to update the weights
optimizer = optim.Adam(model.parameters(), lr=2e-2)

# Loss function: binary cross entropy with logits
# It restricts the predictions to a number between 0 and 1 using the logit function
# then use the bound scarlar to calculate the loss using binary cross entropy
criterion = nn.BCEWithLogitsLoss()

# Use GPU
model = model.to(device)
criterion = criterion.to(device)

# Helper functions

def batch_accuracy(predictions, label):
    """
    Returns accuracy per batch.

    predictions - float
    label - 0 or 1
    """

    # Round predictions to the closest integer using the sigmoid function
    preds = torch.round(torch.sigmoid(predictions))
    # If prediction is equal to label
    correct = (preds == label).float()
    # Average correct predictions
    accuracy = correct.sum() / len(correct)

    return accuracy

def timer(start_time, end_time):
    """
    Returns the minutes and seconds.
    """

    time = end_time - start_time
    mins = int(time / 60)
    secs = int(time - (mins * 60))

    return mins, secs

def train(model, iterator, optimizer, criterion):
    """
    Function to evaluate training loss and accuracy.

    iterator - train iterator
    """
    
    # Cumulated Training loss
    training_loss = 0.0
    # Cumulated Training accuracy
    training_acc = 0.0
    
    # Set model to training mode
    model.train()
    
    # For each batch in the training iterator
    for batch in iterator:
        
        # 1. Zero the gradients
        optimizer.zero_grad()
        
        # batch.text is a tuple (tensor, len of seq)
        text, text_lengths = batch.text
        
        # 2. Compute the predictions
        predictions = model(text, text_lengths).squeeze(1)
        
        # 3. Compute loss
        loss = criterion(predictions, batch.label)
        
        # Compute accuracy
        accuracy = batch_accuracy(predictions, batch.label)
        
        # 4. Use loss to compute gradients
        loss.backward()
        
        # 5. Use optimizer to take gradient step
        optimizer.step()
        
        training_loss += loss.item()
        training_acc += accuracy.item()
    
    # Return the loss and accuracy, averaged across each epoch
    # len of iterator = num of batches in the iterator
    return training_loss / len(iterator), training_acc / len(iterator)

def evaluate(model, iterator, criterion):
    """
    Function to evaluate the loss and accuracy of validation and test sets.

    iterator - validation or test iterator
    """
    
    # Cumulated Training loss
    eval_loss = 0.0
    # Cumulated Training accuracy
    eval_acc = 0
    
    # Set model to evaluation mode
    model.eval()
    
    # Don't calculate the gradients
    with torch.no_grad():
    
        for batch in iterator:

            text, text_lengths = batch.text
            
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            accuracy = batch_accuracy(predictions, batch.label)

            eval_loss += loss.item()
            eval_acc += accuracy.item()
        
    return eval_loss / len(iterator), eval_acc / len(iterator)

# Number of epochs
NUM_EPOCHS = 5

# Lowest validation lost
best_valid_loss = float('inf')

for epoch in range(NUM_EPOCHS):

    start_time = time.time()
    
    # Evaluate training loss and accuracy
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    # Evaluate validation loss and accuracy
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    mins, secs = timer(start_time, end_time)
    
    # At each epoch, if the validation loss is the best
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        # Save the parameters of the model
        torch.save(model.state_dict(), 'model-small.pt')

    print("Epoch {}:".format(epoch+1))
    print("\t Total Time: {}m {}s".format(mins, secs))
    print("\t Train Loss {} | Train Accuracy: {}%".format(round(train_loss, 2), round(train_acc*100, 2)))
    print("\t Validation Loss {} | Validation Accuracy: {}%".format(round(valid_loss, 2), round(valid_acc*100, 2)))

# Load the model with the best validation loss
model.load_state_dict(torch.load('model-small.pt'))

# Evaluate test loss and accuracy
test_loss, test_acc = evaluate(model, test_iterator, criterion)

nlp = spacy.load('en_core_web_sm')

def predict(model, text, tokenized=True):
    """
    Given a tweet, predict the sentiment.

    text - a string or a a list of tokens
    tokenized - True if text is a list of tokens, False if passing in a string
    """

    # Sets the model to evaluation mode
    model.eval()

    if tokenized == False:
        # Tokenizes the sentence
        tokens = [token.text for token in nlp.tokenizer(text)]
    else:
        tokens = text

    # Index the tokens by converting to the integer representation from the vocabulary
    indexed_tokens = [TEXT.vocab.stoi[t] for t in tokens]
    # Get the length of the text
    length = [len(indexed_tokens)]
    # Convert the indices to a tensor
    tensor = torch.LongTensor(indexed_tokens).to(device)
    # Add a batch dimension by unsqueezeing
    tensor = tensor.unsqueeze(1)
    # Converts the length into a tensor
    length_tensor = torch.LongTensor(length)
    # Convert prediction to be between 0 and 1 with the sigmoid function
    prediction = torch.sigmoid(model(tensor, length_tensor))

    # Return a single value from the prediction
    return prediction.item()

# Example prediction from the test set

# List to append data to
# d = []


# for idx in range(10):

#     # Detokenize the tweets from the test set
#     tweet = TreebankWordDetokenizer().detokenize(test_data[idx].text)
                                                 
#     # Append tweet, prediction, and true label
#     d.append({'Tweet': tweet, 'Prediction': predict(model, test_data[idx].text), 'True Label': test_data[idx].label})

# # Convert list to dataframe
# pd.DataFrame(d)
