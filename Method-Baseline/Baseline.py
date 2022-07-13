# Reference: https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Multi%20label%20text%20classification.ipynb

import regex as re
import xlwings as xws
import string

import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

import matplotlib
import matplotlib.pyplot as plt
import statistics

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from tabulate import tabulate

import gensim
from gensim.parsing.preprocessing import STOPWORDS      # List of English Stopwords

import spacy
from nltk.util import ngrams
from nltk.metrics.distance import jaccard_distance
from reynir import Greynir
from reynir_correct import check_single

eng_spacy = spacy.load('en_core_web_sm')
isk_greynir = Greynir()

# Stopwords from: https://github.com/atlijas/icelandic-stop-words/blob/master/all_stop_words.txt
isk_stop = []       # List of Icelandic Stopwords
with open('../lexicons/isk_stop.txt', encoding='utf-8') as f:
    for line in f:
        isk_stop.append(line.strip())

CATEGORIES = ['positive', 'negative', 'neutral']

def lemmatize_eng(input):
    return " ".join([token.lemma_ for token in eng_spacy(input) if token.text not in string.punctuation])

def lemmatize_isk(input):
    token_list = []
    job = isk_greynir.submit(input)

    for sent in job:
        sent.parse()
        if sent.tree is None:
            for t in sent.tokens:
                token_list.append(t.txt)
        else:
            token_list.extend(sent.lemmas)

    return " ".join([token for token in token_list if token not in string.punctuation])

def init(file_name, sheet_name, lang):
    """
    init function launches a password-protected excel file for the user to open and changes it into a dataframe
    all other columns exclusing 'id', 'answer_freetext_value', 'sentiment' is eliminated from the dataframe
    responses in 'answer_freetext_value' is lemmatized
    'sentiment' column is separated into three boolean columns: 'Positive', 'Negative', 'Neutral'

    : param file_name: location/name of the excel file to open
    : param sheet_name: name of the sheet to open

    : return: data in pandas dataframe
    """

    wb = xws.Book(file_name)
    sheet = wb.sheets[sheet_name].used_range

    df = sheet.options(pd.DataFrame, index=False, header=True).value

    header = list(df.columns)

    # Leave only id, freetext, Sentiment column
    to_leave = ['id', 'answer_freetext_value', 'Sentiment']

    for h in header:
        if h not in to_leave:
            del df[h]

    # Lowercase & Strip trailing blanks for sentiment
    df['Sentiment'] = df['Sentiment'].str.lower().str.strip()

    # Drop all rows with either empty freetext or empty sentiment
    df.dropna(subset = ['answer_freetext_value'], inplace=True)
    df.dropna(subset = ['Sentiment'], inplace=True)

    pos = df['Sentiment'].str.contains('positive', regex=False).astype(int)
    neg = df['Sentiment'].str.contains('negative', regex=False).astype(int)
    neu = df['Sentiment'].str.contains('neutral', regex=False).astype(int)

    df['Positive'], df['Negative'], df['Neutral'] = [pos, neg, neu]
    del df['Sentiment']

    if lang == "EN":
        df['answer_freetext_value'] = df['answer_freetext_value'].apply(lambda x: lemmatize_eng(x))
    elif lang == "IS":
        df['answer_freetext_value'] = df['answer_freetext_value'].apply(lambda x: lemmatize_isk(x))

    return df
   
def initialize_pipelines(stop):
    """
    initialize_pipelines function initializes the three basic pipelines: naive bayes, linear SVC, and logistic regression
    
    : param stop: list of stopwords

    : return: NaiveBayes Pipeline, Linear SVC Pipeline, Logistic Regression Pipeline

    """
    NB = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words = stop)),
            ('clf', OneVsRestClassifier(MultinomialNB(
                fit_prior=True, class_prior=None)))
            ])

    SVC = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words = stop)),
            ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1))
            ])

    LogReg = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words = stop)),
            ('clf', OneVsRestClassifier(LogisticRegression(
                solver='sag'), n_jobs=1))
            ])

    return NB, SVC, LogReg

def accuracy(df_actual, df_prediction):
    """
    accuracy function calculates the precision, recall, f1 general, f1 microaverage, f1 macroaverage value
    values that require zero division is set to return 0 as its value

    : param df_actual: dataframe of true values
    : param df_prediction: dataframe of predicted values

    : return [precision, recall, f1_gen]: precision, recall, f1_general in 3 by 3 np array format
    : return [f1_micro, f1_macro]: f1 microaverage and f1 macroaverage in list format
    """
    actual = df_actual
    prediction = df_prediction

    # Calculate precision, recall, accuracy
    precision = precision_score(actual, prediction, average=None, zero_division = 0)
    recall = recall_score(actual, prediction, average=None, zero_division = 0)

    # Calculate f1 score
    f1_gen = f1_score(actual, prediction, average=None, zero_division = 0)
    # Micro average f1 -> calculates positive and negative values globally
    f1_micro = f1_score(actual, prediction, average='micro', zero_division = 0)
    # Macro average f1 -> takes the average of each class's F1 score
    f1_macro = f1_score(actual, prediction, average='macro', zero_division = 0)

    return precision, recall, f1_gen, f1_micro, f1_macro

def display(results):
    """
    display function shows the results in a table

    : param scores: 3 by 3 np array with Sentiments as columns and [Precision, Recall, F1] as indexes
    : param f1s: list containing f1 microaverage and f1 macroaverage
    """
    precisions = []
    recalls = []
    f1_gens = []
    f1_micros = []
    f1_macros = []

    for items in results:
        precisions.append(items[0])
        recalls.append(items[1])
        f1_gens.append(items[2])
        f1_micros.append(items[3])
        f1_macros.append(items[4])

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


def classify(train, test, pipeline):
    """
    classify function trains the pipeline with the provided training data, and predicts the result for the testing data
    function calls accuracy function to compare the actual data and predicted result

    : param train: dataframe of training data
    : param test: dataframe of testing data
    : param pipeline: initialized pipeline of given classifier (naive bayes, linear SVC, logistic regression)

    : return: np.array consisting precision, recall, f1_gen data and list containing f1 microaverage and f1 macroaverage
    """
    actual = test[CATEGORIES]
    prediction = pd.DataFrame(columns=CATEGORIES)

    for category in CATEGORIES:
        pipeline.fit(train.answer_freetext_value, train[category])
        predicted = pipeline.predict(test.answer_freetext_value)

        prediction[category] = pd.Series(predicted)

    return accuracy(actual, prediction)

def baseline(df, lang):
    """
    baseline function sets the stopwords based on the input language, and divides the input df into 5 sections that will be used for training/testing
    function calls classfiy function to perform pipeline fit
    returned accuracy value is averaged to given an average accuracy score
    display function is called to display the result in tabulated format

    : param df: input dataframe (with 'id', 'answer_freetext_value', 'Positive', 'Negative', 'Neutral' as columns)
    : param lang: language of input data (either "EN" or "IS")
    """
    if lang == "EN":
        stop = STOPWORDS
    elif lang == "IS":
        stop = isk_stop

    [NB, SVC, LogReg] = initialize_pipelines(stop)

    kf = KFold(n_splits=5, random_state=99, shuffle=True)
    num_split = kf.get_n_splits(df)

    scores_NB = np.array([0, 0, 0])
    scores_SVC = np.array([0, 0, 0])
    scores_LogReg = np.array([0, 0, 0])

    f1_NB = [0, 0]
    f1_SVC = [0, 0]
    f1_LogReg = [0, 0]

    results_NB = []
    results_SVC = []
    results_LogReg = []

    for train_index, test_index in kf.split(df):
        train = df.iloc[train_index]
        test = df.iloc[test_index]

        result = classify(train, test, NB)
        results_NB.append(result)

        result = classify(train, test, SVC)
        results_SVC.append(result)
        
        result = classify(train, test, LogReg)
        results_LogReg.append(result)

    print(lang)
    print("Naive Bayes")
    display(results_NB)

    print("\n\nLinear SVC")
    display(results_SVC)

    print("\n\nLogistic Regression")
    display(results_LogReg)

    print("\n\n\n\n")
