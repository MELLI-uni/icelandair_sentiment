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

import transformers
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import logging

device = 'cuda' if cuda.is_available() else 'cpu'
logging.set_verbosity_warning()
logging.set_verbosity_error()

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 4

LEARNING_RATE = 1e-05
CATEGORIES = ['Positive', 'Negative', 'Neutral']

encoding_dict = {
                'positive':0,
                'neutral':2,
                'negative':1,
                'negativa':1
                }

eng_negating = r'\s[Bb]ut\.*,*\s|\s[Hh]owever\.*,*\s'
isk_negating = r'\s[Ee]n\.*,*\s|\s[Nn]ema\.*,*\s'

def init(file_name, sheet_name, lang):
    wb = xws.Book(file_name)
    sheet = wb.sheets[sheet_name].used_range

    df = sheet.options(pd.DataFrame, index=False, header=True).value

    header = list(df.columns)

    to_leave = ['answer_freetext_value', 'Sentiment']

    for h in header:
        if h not in to_leave:
            del df[h]

    df['Sentiment'] = df['Sentiment'].str.lower().str.strip()

    df.dropna(subset = ['answer_freetext_value'], inplace=True)
    df.dropna(subset = ['Sentiment'], inplace=True)

    return df

def combine_df(df1, df2):
    df = pd.concat([df1, df2], ignore_index=True)

    return df

def separate_multi(df, lang):
    """
    separate_multi function separates rows of dataframe with multiple sentiments
    Separation Rule:
        1. if there is a new line, separate at new line
        2. if there is a negating conjugation, separate at negating conjugation
        3. if there is a period, separate at period
        4. if there is a comma, separate at comma
        else. delete row from dataframe
    """

    df['Sentiment'] = list(df['Sentiment'].str.split(r'\W+'))

    df_multi = df.loc[(df['Sentiment']).str.len() > 1]
    df.drop(df_multi.index, inplace=True)
    df = df.explode(['Sentiment'])

    df_multi['ct_senti'] = (df_multi['Sentiment']).str.len()

    df_multi['ct_sep'] = (df_multi['answer_freetext_value']).str.split(r'\n+').str.len()
    df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep']), 'answer_freetext_value'] = (df_multi['answer_freetext_value']).str.split(r'\n+')
    df_sep = df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep'])]
    df_multi.drop(df_sep.index, inplace=True)
    del df_multi['ct_sep']

    if lang == 'EN':
        negating_pattern = eng_negating
    elif lang == 'IS':
        negating_pattern = isk_negating
    
    df_multi['ct_sep'] = (df_multi['answer_freetext_value']).str.split(negating_pattern).str.len()
    df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep']), 'answer_freetext_value'] = (df_multi['answer_freetext_value']).str.split(negating_pattern)
    df_temp = df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep'])]
    df_sep = pd.concat([df_sep, df_temp], sort=False)
    df_multi.drop(df_temp.index, inplace=True)
    del df_multi['ct_sep']

    df_multi['ct_sep'] = (df_multi['answer_freetext_value']).str.strip(r'[.!?]').str.split(r'[.!?]').str.len()
    df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep']), 'answer_freetext_value'] = (df_multi['answer_freetext_value']).str.strip(r'[.!?]').str.split(r'[.!?]')
    df_temp = df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep'])]
    df_sep = pd.concat([df_sep, df_temp], sort=False)
    df_multi.drop(df_temp.index, inplace=True)
    del df_multi['ct_sep']

    df_multi['ct_sep'] = (df_multi['answer_freetext_value']).str.split(r'[,;]').str.len()
    df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep']), 'answer_freetext_value'] = (df_multi['answer_freetext_value']).str.split(r'[,;]')
    df_temp = df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep'])]
    df_sep = pd.concat([df_sep, df_temp], sort=False)
    df_multi.drop(df_temp.index, inplace=True)
    del df_multi['ct_sep']
    
    del df_sep['ct_senti']
    del df_sep['ct_sep']
    df_sep = df_sep.explode(['Sentiment', 'answer_freetext_value'])

    df = pd.concat([df, df_sep], ignore_index=True, sort=False)

    return df

def sentiment_mapping(df):
    df['Sentiment'] = df.Sentiment.map(encoding_dict)

    # Line will be deleted later
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
                max_length = self.max_len,
                return_token_type_ids=True
        )

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
    #test_size = 0.2
    #df_train, df_test = train_test_split(df, test_size=0.2, random_state=99)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    #df_train.to_pickle('./train.pkl')
    #df_test.to_pickle('./test.pkl')

    training_set = SentimentData(df_train, tokenizer, MAX_LEN)
    testing_set = SentimentData(df_test, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': TEST_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    return training_loader, testing_loader

def accuracy(list_actual, list_prediction):
    actual = list_actual
    prediction = list_prediction

    precision = precision_score(actual, prediction, average=None, zero_division = 0).tolist()
    recall = recall_score(actual, prediction, average=None, zero_division = 0).tolist()
 
    f1_gen = f1_score(actual, prediction, average=None, zero_division = 0).tolist()
    f1_micro = f1_score(actual, prediction, average='micro', zero_division = 0)
    f1_macro = f1_score(actual, prediction, average='macro', zero_division = 0)

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

    #score_compile = scores
    df_score = pd.DataFrame(data=scores, index=['Precision', 'Recall', 'F1'], columns=CATEGORIES)
    df_average = pd.DataFrame(data=f1s, index=['F1 Microaverage', 'F1 Macroaverage'], columns=['Scores'])

    print(tabulate(df_score, headers='keys', tablefmt='pretty'))
    print(tabulate(df_average, headers='keys', tablefmt='pretty'))

def calculate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()

    return n_correct

class RobertaClass(torch.nn.Module):
    def __init__(self, model_name):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)

        return output

class XLMRobertaClass(torch.nn.Module):
    def __init__(self, model_name):
        super(XLMRobertaClass, self).__init__()
        self.l1 = XLMRobertaModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)

        return output

def tuning(tokenizer, model, tune_file, save_path):
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=tune_file,
        block_size=512,
        )

    data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
            )

    training_args = TrainingArguments(
            output_dir=save_path,
            overwrite_output_dir=True,
            num_train_epochs=25,
            per_device_train_batch_size=48,
            save_steps=500,
            save_total_limit=2,
            seed=1,
            )

    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset
            )

    trainer.train()
    trainer.save_model(save_path)

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
        targets  = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim = 1)
        n_correct += calculate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _%5000==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct * 100)/nb_tr_examples

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
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_accuracy(big_idx, targets)

            #print(outputs.data)

            predicted.extend(big_idx.tolist())
            actual.extend(targets.tolist())

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct * 100)/nb_tr_examples

                print(f"Validation Loss per 100 Steps: {loss_step}")
                print(f"Validation Accuracy per 100 Steps: {accu_step}")

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples

    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return accuracy(actual, predicted)

def test_vanilla_5fold(df, lang):
    kf = KFold(n_splits=5, random_state=99, shuffle=True)
    num_split = kf.get_n_splits(df)

    precisions = []
    recalls = []
    f1_gens = []
    f1_micros = []
    f1_macros = []

    # Create loss function
    loss_function = torch.nn.CrossEntropyLoss()

    for train_index, test_index in kf.split(df):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]

        # Tuned roBERTa model initilization
        if lang == "EN":
            vanilla_model = RobertaClass('roberta-base')
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True, max_length = MAX_LEN)
        elif lang == "IS":
            vanilla_model = RobertaClass('mideind/IceBERT')
            tokenizer = RobertaTokenizer.from_pretrained('mideind/IceBERT', truncation=True, do_lower_case=True, max_length = MAX_LEN)
        vanilla_model.to(device)

        # Create optimizer
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params = vanilla_model.parameters(), lr=LEARNING_RATE)

        # Load data
        training_loader, testing_loader = data_loading(df_train, df_test, tokenizer)

        # Train model
        EPOCHS = 1
        for epoch in range(EPOCHS):
            train(vanilla_model, training_loader, epoch, loss_function, optimizer)

        # Validate model
        precision, recall, f1_gen, f1_micro, f1_macro = valid(vanilla_model, testing_loader, loss_function)

        precisions.append(precision)
        recalls.append(recall)
        f1_gens.append(f1_gen)
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)

    print("VANILLA MODEL for", lang.upper())
    display(precisions, recalls, f1_gens, f1_micros, f1_macros)

def test_tuning_5fold(df, lang):
    kf = KFold(n_splits=5, random_state=99, shuffle=True)
    num_split = kf.get_n_splits(df)

    precisions = []
    recalls = []
    f1_gens = []
    f1_micros = []
    f1_macros = []

    # Create loss function
    loss_function = torch.nn.CrossEntropyLoss()

    for train_index, test_index in kf.split(df):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]

        # Tuned roBERTa model initilization
        if lang == "EN":
            tuned_model = RobertaClass('ModelTuning/roberta-retrained')
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True, max_length = MAX_LEN)
        elif lang == "IS":
            tuned_model = RobertaClass('ModelTuning/icebert-retrained')
            tokenizer = RobertaTokenizer.from_pretrained('mideind/IceBERT', truncation=True, do_lower_case=True, max_length = MAX_LEN)
        tuned_model.to(device)

        # Create optimizer
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params = tuned_model.parameters(), lr=LEARNING_RATE)

        # Load data
        training_loader, testing_loader = data_loading(df_train, df_test, tokenizer)

        # Train model
        EPOCHS = 1
        for epoch in range(EPOCHS):
            train(tuned_model, training_loader, epoch, loss_function, optimizer)

        # Validate model
        precision, recall, f1_gen, f1_micro, f1_macro = valid(tuned_model, testing_loader, loss_function)

        precisions.append(precision)
        recalls.append(recall)
        f1_gens.append(f1_gen)
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)

    print("TUNED MODEL for", lang.upper())
    display(precisions, recalls, f1_gens, f1_micros, f1_macros)

def test_xlm_vanilla_5fold(df):
    kf = KFold(n_splits=5, random_state=99, shuffle=True)
    num_split = kf.get_n_splits(df)

    precisions = []
    recalls = []
    f1_gens = []
    f1_micros = []
    f1_macros = []

    # Create loss function
    loss_function = torch.nn.CrossEntropyLoss()

    for train_index, test_index in kf.split(df):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]

        vanilla_model = XLMRobertaClass('xlm-roberta-base')
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', truncation=True, do_lower_case=True, max_length = MAX_LEN)
        vanilla_model.to(device)

        # Create optimizer
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params = vanilla_model.parameters(), lr=LEARNING_RATE)

        # Load data
        training_loader, testing_loader = data_loading(df_train, df_test, tokenizer)

        # Train model
        EPOCHS = 1
        for epoch in range(EPOCHS):
            train(vanilla_model, training_loader, epoch, loss_function, optimizer)

        # Validate model
        precision, recall, f1_gen, f1_micro, f1_macro = valid(vanilla_model, testing_loader, loss_function)

        precisions.append(precision)
        recalls.append(recall)
        f1_gens.append(f1_gen)
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)

    display(precisions, recalls, f1_gens, f1_micros, f1_macros)

def test_xlm_tuning_5fold(df):
    kf = KFold(n_splits=5, random_state=99, shuffle=True)
    num_split = kf.get_n_splits(df)

    precisions = []
    recalls = []
    f1_gens = []
    f1_micros = []
    f1_macros = []

    # Create loss function
    loss_function = torch.nn.CrossEntropyLoss()

    for train_index, test_index in kf.split(df):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]

        vanilla_model = XLMRobertaClass('ModelTuning/isk_xlm_roberta-retrained')
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', truncation=True, do_lower_case=True, max_length = MAX_LEN)
        vanilla_model.to(device)

        # Create optimizer
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params = vanilla_model.parameters(), lr=LEARNING_RATE)

        # Load data
        training_loader, testing_loader = data_loading(df_train, df_test, tokenizer)

        # Train model
        EPOCHS = 1
        for epoch in range(EPOCHS):
            train(vanilla_model, training_loader, epoch, loss_function, optimizer)

        # Validate model
        precision, recall, f1_gen, f1_micro, f1_macro = valid(vanilla_model, testing_loader, loss_function)

        precisions.append(precision)
        recalls.append(recall)
        f1_gens.append(f1_gen)
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)

    display(precisions, recalls, f1_gens, f1_micros, f1_macros)
