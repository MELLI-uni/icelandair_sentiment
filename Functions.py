from tokenize import String
from itertools import groupby
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

import re
import xlwings as xws

import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS
STOPWORDS = STOPWORDS.union(set(['icelandair']))

import spacy

eng_conjunction = {"but", "conversly", "however"}
isk_conjuction = {}

# Dataframe Initialization
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

# Data Cleaning
def clean(df):
    """
    clean function eliminates columns that are not needed, rows with no freetext or sentiment, lowercase and strips
    trailing blank spaces in the sentiment column, and separates the sentiment into three boolean columns (pos, neg, neu)

    : param df: panda dataframe

    : return: panda dataframe with [id(int), freetext(String), Positive(bool), Negative(bool), Neutral(bool)]
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

# TASK1-placeholder
# TODO: TASK1-B3-extend
def clean_multi(df, lang):
    """
    clean function eliminates columns that are not needed, rows with no freetext or sentiment, lowercase and strips
    trailing blank spaces in the sentiment column, and separates the sentiment into three boolean columns (pos, neg, neu)

    : param df: panda dataframe

    : return: panda dataframe with [id(int), freetext(String), Positive(bool), Negative(bool), Neutral(bool)]

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

    # Find all rows with multiple sentiments and store it to separate dataframe
    df_temp = df.loc[(df['Sentiment']).str.len() > 1]
    df.drop(df_temp.index, inplace=True)

    print("\nMultiple Sentiment: " + str(len(df_temp.index)))

    df_count = df_temp[['Sentiment', 'answer_freetext_value']].copy()   # Create a copy of the dataframe with multiple sentiments
    df_count['answer_freetext_value'] = df_count['answer_freetext_value'].str.replace(r'(\s*[Tt]hank [Yy]ou)\.\s*','', regex=True)  # Replace all 'Thank you.' to blank space
    
    df_count['Sentiment'] = df_count['Sentiment'].str.len() # Count number of sentiments
    df_count['New Line'] = df_count['answer_freetext_value'].str.split(r'\n+').str.len()    # Count number of sentences separated by new line
    df_count['Period'] = df_count['answer_freetext_value'].str.strip('\.$').str.split(r'(?<=[a-zA-Z])\.').str.len() # Count number of sentences separated by period
    # Reference for lookbehind https://www.geeksforgeeks.org/python-regex-lookbehind/

    df_count['Elim'] = df_temp['Sentiment'].apply(lambda x: [i[0] for i in groupby(x)]).str.len()   # Remove consecutive duplicates in sentiment list
    #df_count['Word'] = df_count['answer_freetext_value'].str.split().str.len()
    #df_count['Else'] = df_count['answer_freetext_value'].str.slice()

    df_count.loc[(df_count['Sentiment'] == df_count['New Line']), 'Type'] = 'New Line'
    df_count.loc[(df_count['Sentiment'] == df_count['Period']) & (df_count['Type'].isnull()), 'Type'] = 'Period'

    df_count.loc[(df_count['Elim'] == df_count['New Line']), 'Type'] = 'New Line'
    df_count.loc[(df_count['Elim'] == df_count['New Line']), 'Sentiment'] = df_count['Elim']    # Replace sentiment label with duplicate removed list
    del df_count['Elim']
    #df_count.loc[(df_count['Sentiment'] == df_count['Comma']) & (df_count['Type'].isnull()), 'Type'] = 'Comma'

    #df_temp.to_excel("compare.xlsx")
    df_count.to_excel("text.xlsx")

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

    #t0 = time.time()

    eng_NE = spacy.load('en_core_web_sm')

    df['Change'] = df['answer_freetext_value'].apply(lambda x: [str(ent.lemma_) for ent in eng_NE(x) if 
                                                                (not ent.ent_type_ 
                                                                and not re.findall("\s", str(ent.text)) 
                                                                and str(ent.text).lower() not in STOPWORDS)
                                                                ])
    del df['answer_freetext_value']
    
    #df['Change'] = df['answer_freetext_value'].apply(lambda x: " ".join([ent.text for ent in nlp(x) if not ent.ent_type_]))

    #df['Change'] = df['Change'].apply(lambda x: [item for item in x.split() if item.lower() not in STOPWORDS])
    #del df['answer_freetext_value']
    #print(df)
    #text = "Her check in Frankfurt in bad mood: Check in Frankfurt very unfriendly! No upgrade available inspire  of  information on board and by service center! The young man told me that there is no upgrade for 2 of my children available, because he cannot handle any payment. As multi flying passenger I was very much astonished! Big complain of mine (3th time with icelandair business  5 persons!)"

    #text_no_namedentities = [str(ent.text) for ent in document if not ent.ent_type_]

    #text_no_namedentities = []
    # filtered_sentence =  []
    # for ent in eng_NE(text):
    #     token = str(ent.text)
    #     if not ent.ent_type_ and not re.findall("\s", token):
    #         if token.lower() not in STOPWORDS:
    #             filtered_sentence.append(str(ent.lemma_))

    #filtered_sentence = remove_stopwords(text_no_namedentities)

    # print(filtered_sentence)

    # print("Run Time: " + str(time.time() - t0))

    return df