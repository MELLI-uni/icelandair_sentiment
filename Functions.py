from tokenize import String
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

import re
import xlwings as xws

import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS

import spacy

# Open xlsx file
# Reference: https://davidhamann.de/2018/02/21/read-password-protected-excel-files-into-pandas-dataframe/
def init(file_name, sheet_name):
    """
    init function launches a password-protected excel file for the user to open and changes it into a datafram

    : param file_name: location/name of the excel file to open
    : param sheet_name: name of the sheet to open

    : return: data in pandas dataframe
    """

    wb = xws.Book(file_name)
    sheet = wb.sheets[sheet_name].used_range

    df = sheet.options(pd.DataFrame, index=False, header=True).value
    return(df)

# Eliminate columns not needed
# LowerCase & Strip trailing blank space in Sentiment
# Eliminate rows with no freetext or Sentiment
# Sentiment into Boolean column of Positive, Negative, Neutral
def clean(df):
    """
    input: panda_df
    output: cleaned panda_df with (id, freetext, Sentiment)
    """

    header = list(df.columns)

    # Leave only id, freetext, Sentiment column
    to_leave = ['id', 'answer_freetext_value', 'Sentiment']
    #to_leave = ['id', 'Sentiment']  #temp

    for h in header:
        if h not in to_leave:
            del df[h]

    # Lowercase & Strip trailing blanks for sentiment
    df['Sentiment'] = df['Sentiment'].str.lower().str.strip()

    # Drop all rows with either empty freetext or emtpy sentiment
    df.dropna(subset = ['answer_freetext_value'], inplace=True)
    df.dropna(subset = ['Sentiment'], inplace=True)
    sentiment_li = ['positive', 'negative', 'neutral']
    df = df[df['Sentiment'].isin(sentiment_li)]

    # Create new column of Positive, Negative, Neutral Boolean
    pos = (df['Sentiment'] == 'positive').astype(int)
    neg = (df['Sentiment'] == 'negative').astype(int)
    neu = (df['Sentiment'] == 'neutral').astype(int)
    df['Positive'], df['Negative'], df['Neutral'] = [pos, neg, neu]
    del df['Sentiment']

    return df

def clean_multi(df):
    """
    input: panda_df
    output: cleaned panda_df with (id, freetext, Sentiment)
    """

    header = list(df.columns)

    # Leave only id, freetext, Sentiment column
    to_leave = ['id', 'answer_freetext_value', 'Sentiment']

    for h in header:
        if h not in to_leave:
            del df[h]

    # Lowercase & Strip trailing blanks for sentiment
    # Change sentiment column into list by spliting at non-word component
    df['Sentiment'] = df['Sentiment'].str.lower().str.strip()
    df['Sentiment'] = list(df['Sentiment'].str.split(r'\W+'))

    # Drop all rows with either empty freetext or emtpy sentiment
    df.dropna(subset = ['answer_freetext_value'], inplace=True)
    df.dropna(subset = ['Sentiment'], inplace=True)

    # Create new column of Positive, Negative, Neutral Boolean
    pos = df['Sentiment'].str.contains('positive', regex=False).astype(int)
    neg = df['Sentiment'].str.contains('negative', regex=False).astype(int)
    neu = df['Sentiment'].str.contains('neutral', regex=False).astype(int)
    df['Positive'], df['Negative'], df['Neutral'] = [pos, neg, neu]
    del df['Sentiment']

    return df

# Text preprocessing
def process(df):
    """
    process Function eliminates stopwords and Named Entities, tokenizes and lemmatizes the given sentence

    : param df: dataframe where text processing is needed

    : return: Pandas dataframe with processed free-text
    """

    # Eliminate named entities and join to one string
    nlp = spacy.load('en_core_web_sm')
    df['Change'] = df['answer_freetext_value'].apply(lambda x: " ".join([ent.text for ent in nlp(x) if not ent.ent_type_]))

    df['Change'] = df['Change'].apply(lambda x: [item for item in x.split() if item.lower() not in STOPWORDS])
    del df['answer_freetext_value']
    print(df)
    #text = "Her check in Frankfurt in bad mood: Check in Frankfurt very unfriendly! No upgrade available inspire  of  information on board and by service center! The young man told me that there is no upgrade for 2 of my children available, because he cannot handle any payment. As multi flying passenger I was very much astonished! Big complain of mine (3th time with icelandair business  5 persons!)"
    #filtered_sentence = remove_stopwords(text)

    # Eliminate named entities
    # nlp = spacy.load('en_core_web_sm')
    # document = nlp(text)

    
    # text_no_namedentities = [str(ent.text) for ent in document if not ent.ent_type_]
    # filtered_sentence = remove_stopwords(text_no_namedentities)
    # print(filtered_sentence)