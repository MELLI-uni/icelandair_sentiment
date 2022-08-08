import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import random

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tabulate import tabulate
from tqdm impor tqdm

#SEED = 99

#torch.manual_seed(SEED)
#torch.backends.cudnn.deterministic = True

#TEXT = data.Field(tokenize='spacy', tokenizer_language = 'en_core_web_sm')
#LABEL = data.LabelField()

#train_data, test_data = datasets.TREC.splits(TEXT, LABEL, fine_grained=False)

#train_data, valid_data = train_data.split(random_state=random.seed(SEED))

#df_eng = pd.read_pickle('../Data/eng_total.pkl')

def sentiment_mapping(df):
    df['Sentiment'] = df.Sentiment.map(encoding_dict)

    del df['id']

    return df

class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.answer_freetext_value
        self.targets = dataframe.Sentiment
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_token_type_ids=True)

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.targets[index], dtype=torch.float)
                }

def data_loading(df_train, df_test, tokenizer):
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    training_set = SentimentData(df_train, tokenizer, MAX_LEN)
    testing_set = SentimentData(df_test, tokenizer, MAX_LEN)

    train_params = {
            'batch_size': TRAIN_BATCH_SIZE,
            'shuffle':True,
            'num_workers':0}

    test_params = {
            'batch_size': TEST_BATCH_SIZE,
            'shuffle':False,
            'num_workers':0}

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    return training_loader, testing_loader

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
    df_average = pd.DataFrame(data=f1s, index=['F1 Microaverage', 'F1 Macroaverage'], column=['Scores'])

    print(tabulate(df_score, headers='keys', tablefmt='pretty'))
    print(tabulate(df_average, headers='keys', tablefmt='pretty'))

def calculate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()

    return n_correct

class CNN(torch.nn.Module):
    def __init__(self, vocab_size, embeedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = torch.nn.ModuleList([
            nn.Conv2d(in_channels = 1,
                out_channels = n_filters,
                kernel_size = (fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = torch.nn.Linear(len(filter_sizes) * n_filters, output_dims)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0)
        
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)

        conved = [torch.nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        
        pooled = [torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim = 1))
        
        return self.fc(cat)
    
def train(model, training_loader, epoch, loss_function, optimizer):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0

    model.train()

    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calculate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _%5000==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples

            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"The Total Accuracy for Epoch {epoch}: {(n_correct * 100)/nb_tr_examples}")

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples

    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

def valid(model, testing_loader, loss_function):
    model.eval()

    n_correct = 0
    n_wrong = 0
    total = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0

    actual = []
    predicted = []

    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_tye_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_accuracy(big_idx, targets)

            predicted.extend(big_idx.tolist())
            actual.extend(targets.tolist())

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples

                print(f"Validation Loss per 100 Steps: {loss_step}")
                print(f"Validation Accuracy per 100 Steps: {accu_step}")

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples

    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    return accuracy(actual, predicted)
