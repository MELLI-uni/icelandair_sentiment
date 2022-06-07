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

eng_pattern = r'\s[Bb]ut\.*,*\s|\s[Hh]owever\.*,*\s|\s[Ee]xcept\.*,*\s'
isk_pattern = r'\s[Ee][n]\.*,*\s|\s[Þþ]ó\.*,*\s|\s[Nn]ema\.*,*\s|\s[Hh]ins vegar\.*,*\s|\sfyrir utan\.*,*\s'

import spacy

# regex pattern for flight names
regex_plane = r'(A3\d{2}(-\d{3})?)|(7\d7(-\d{3})?)|FI\d+'

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
    df = df.explode(['Sentiment'])

    #df_temp.to_excel("multi.xlsx")      # LINE TO DELETE

    df_count = df_temp[['Sentiment', 'answer_freetext_value']].copy()   # Create a copy of the dataframe with multiple sentiments
    df_count['answer_freetext_value'] = df_count['answer_freetext_value'].str.replace(r'\s*([Tt]hank [Yy]ou)\.*\s*','', regex=True)  # Replace all 'Thank you.' to blank space
    df_count['answer_freetext_value'] = df_count['answer_freetext_value'].str.replace(r'\s*[Yy]es\.*\,*\s*','', regex=True)  # Replace all 'Yes.' to blank space

    df_count['answer_freetext_value'] = df_count['answer_freetext_value'].str.replace(r'\s*[Tt]akk\.\s*','', regex=True)
    df_count['answer_freetext_value'] = df_count['answer_freetext_value'].str.replace(r'\s*[Jj]á\.*\,*\s*','', regex=True)
    
    df_count['Sentiment'] = df_count['Sentiment'] # Count number of sentiments
    df_count['New Line'] = df_count['answer_freetext_value'].str.split(r'\n+')    # Count number of sentences separated by new line

    if lang == "IS":
        negating_conjugation = isk_pattern
    else:
        negating_conjugation = eng_pattern
    
    df_count['Negating'] = df_count['answer_freetext_value'].str.split(negating_conjugation)    # Count number of sentences separated by negating conjugations
    # TODO: Investigate with \p{L} -> why is it not workiiinnnggggggg
    df_count['Period'] = df_count['answer_freetext_value'].str.strip(r'\W+').str.split(r'(?<=[a-zA-ZÁáÐðÉéÍíÓóÚúÝýÞþÆæÖö])\.(?=\s*[a-zA-ZÁáÐðÉéÍíÓóÚúÝýÞþÆæÖö])') # Count number of sentences separated by period
    #df_count['Word'] = df_count['answer_freetext_value'].str.split()  # Count number of words in the sentence
    #df_count['Word'] = df_count['Word'].apply(lambda x: [' '.join(i.tolist()) for i in (np.array_split(np.array(x), 2))])
    #df_count['Wordc'] = df_count.apply(lambda x: np.array(x['Word']))

    df_count['Punct'] = df_count['answer_freetext_value'].str.split(r',|;') # Count number of sentences separated by comma or semicolon
    df_count['Elim'] = df_temp['Sentiment'].apply(lambda x: [i[0] for i in groupby(x)])   # Remove consecutive duplicates in sentiment list

    #TODO: Determine priority or if accuracy of program is high enough, train this portion with transformer?!?!
    df_count.loc[(df_count['Sentiment'].str.len() == df_count['New Line'].str.len()), 'Type'] = 'New Line'
    df_count.loc[(df_count['Sentiment'].str.len() == df_count['Negating'].str.len()) & (df_count['Type'].isnull()), 'Type'] = 'Negating'
    df_count.loc[(df_count['Sentiment'].str.len() == df_count['Period'].str.len()) & (df_count['Type'].isnull()), 'Type'] = 'Period'

    df_count.loc[(df_count['Elim'].str.len() == df_count['New Line'].str.len()), 'Type'] = 'New Line'
    df_count.loc[(df_count['Elim'].str.len() == df_count['New Line'].str.len()), 'Sentiment'] = df_count['Elim']    # Replace sentiment label with duplicate removed list

    df_count.loc[(df_count['Elim'].str.len() == df_count['Negating'].str.len()) & (df_count['Type'].isnull()), 'Sentiment'] = df_count['Elim']    # Replace sentiment label with duplicate removed list
    df_count.loc[(df_count['Elim'].str.len() == df_count['Negating'].str.len()) & (df_count['Type'].isnull()), 'Type'] = 'Negating'
    del df_count['Elim']

    df_count.loc[(df_count['Period'].str.len() == 1) & (df_count['Sentiment'].str.len() == df_count['Punct'].str.len()) & (df_count['Type'].isnull()), 'Type'] = 'Punct'

    df_temp['Sentiment'] = df_count['Sentiment']
    df_temp.loc[(df_count['Type'] == 'New Line'), 'answer_freetext_value'] = df_count['New Line'][df_count['Type'] == 'New Line']
    df_temp.loc[(df_count['Type'] == 'Negating'), 'answer_freetext_value'] = df_count['Negating'][df_count['Type'] == 'Negating']
    df_temp.loc[(df_count['Type'] == 'Period'), 'answer_freetext_value'] = df_count['Period'][df_count['Type'] == 'Period']
    df_temp.loc[(df_count['Type'] == 'Punct'), 'answer_freetext_value'] = df_count['Punct'][df_count['Type'] == 'Punct']
    df_temp['Type'] = df_count['Type']
    df_temp.to_excel("multi_det.xlsx")      # LINE TO DELETE

    df_temp.dropna(subset = ['Type'], inplace=True)
    del df_temp['Type']

    df_temp = df_temp.explode(['Sentiment', 'answer_freetext_value'])
    #df_temp.to_excel("multi_explode.xlsx")       # LINE TO DELETE

    df = pd.concat([df, df_temp], ignore_index=True, sort=False)

    # Create new column of Positive, Negative, Neutral Boolean
    # pos = df['Sentiment'].str.contains('positive', regex=False).astype(int)
    # neg = df['Sentiment'].str.contains('negative', regex=False).astype(int)
    # neu = df['Sentiment'].str.contains('neutral', regex=False).astype(int)
    # df['Positive'], df['Negative'], df['Neutral'] = [pos, neg, neu]

    df.loc[df['Sentiment'] == 'positive', 'Sentiment'] = 1
    df.loc[df['Sentiment'] == 'negative', 'Sentiment'] = -1
    df.loc[df['Sentiment'] == 'neutral', 'Sentiment'] = 0
    #del df['Sentiment']

    return df

def load_lexicon():
    flights = open("./lexicons/flight-city.txt", 'r', encoding='utf-8').read()
    flight_lexicon = flights.split()

    return flight_lexicon

# Text preprocessing
def process(df, lang, FLIGHTS):
    """
    process Function eliminates stopwords and Named Entities, tokenizes and lemmatizes the given sentence

    : param df: dataframe where text processing is needed

    : return: Pandas dataframe with processed free-text
    """

    eng_NE = spacy.load('en_core_web_sm')
    #isk_NE = 
    # if lang == "IS":
    #     named_ent = isk_NE
    # else:
    #     named_ent = eng_NE

    df['Change'] = df['answer_freetext_value'].apply(lambda x: [str(ent.lemma_).lower() for ent in eng_NE(x) if 
                                                                (not ent.ent_type_ 
                                                                and not re.findall(r'\W|\d', str(ent.text))
                                                                and not re.findall(r'no|nei|n/a|N/A', str(ent.text).lower())
                                                                and not re.findall(regex_plane, str(ent.text).lower())
                                                                and str(ent.text).lower() not in STOPWORDS
                                                                and str(ent.text) not in FLIGHTS)
                                                                ])

    df = df.explode(['Change'], ignore_index=True)
    df.dropna(subset = ['Change'], inplace=True)

    idx_not = df.index[df['Change'].str.contains('not', regex=True)].tolist()
    idx_follow = [i+1 for i in idx_not]
    #print(idx_follow)
    df.loc[idx_follow, 'Sentiment'] = -df['Sentiment']

    df = df.drop(idx_not)

    # df['Change'] = df['answer_freetext_value'].apply(lambda x: [str(ent.lemma_) for ent in eng_NE(x) if 
    #                                                             (not ent.ent_type_ 
    #                                                             and not re.findall(r'\W', str(ent.text)))
    #                                                             ])

    # lower case everything left
    # delete blank text rows, nei, n/a
    # eliminate those with single letter and all numbers
    # eliminate words from flight-city
    # eliminate all seat combination
    # eliminate flight related
    # icelandair plane names FI[]

    del df['answer_freetext_value']

    df.to_excel("check.xlsx")
    
    return df

def process_sentence(sentence, lang, FLIGHTS):
    """
    process Function eliminates stopwords and Named Entities, tokenizes and lemmatizes the given sentence

    : param df: dataframe where text processing is needed

    : return: Pandas dataframe with processed free-text
    """

    eng_NE = spacy.load('en_core_web_sm')

    cleaning = [str(ent.lemma_).lower() for ent in eng_NE(sentence) if
                (not ent.ent_type_
                and (ent.pos_ == "ADJ")
                and not re.findall(r'\W|\d', str(ent.text))
                and not re.findall(r'no|nei|n/a|N/A', str(ent.text).lower())
                and not re.findall(regex_plane, str(ent.text).lower())
                and str(ent.text).lower() not in STOPWORDS
                and str(ent.text) not in FLIGHTS)
                ]

    # cleaning = [str(ent.lemma_).lower() for ent in eng_NE(sentence) if
    #             (not ent.ent_type_
    #             and str(ent.lemma_).lower() != "flight"
    #             and not re.findall(r'\W|\d', str(ent.text))
    #             and not re.findall(r'no|nei|n/a|N/A', str(ent.text).lower())
    #             and not re.findall(regex_plane, str(ent.text).lower())
    #             and str(ent.text).lower() not in STOPWORDS
    #             and str(ent.text) not in FLIGHTS)
    #             ]


    # cleaning = []
    # negate = ""
    # mult = ""
    # for ent in eng_NE(sentence):
    #     token = ""
    #     if (str(ent) == "not"):
    #         negate = "-"
    #         continue
    #     if ((ent.pos_ == 'ADJ') | (ent.pos_ == 'ADV')):
    #         mult = "10"
    #     if (not ent.ent_type_
    #         and not re.findall(r'\W|\d', str(ent.text))
    #         and not re.findall(r'no|nei|n/a|N/A', str(ent.text).lower())
    #         and not re.findall(regex_plane, str(ent.text).lower())
    #         and str(ent.text).lower() not in STOPWORDS
    #         and str(ent.text) not in FLIGHTS):
    #         token = negate + mult + str(ent.lemma_).lower()
    #         negate = ""
    #         mult = ""

    #     cleaning.append(token)
    
    return cleaning

# Intensify adjectives and adverbs when calculating scores

# Deal with not
# Filter out not from the lexicon list (when developing lexicon list)
# If not is placed in front of an adjective then change the storing sentiment when developing lexicon list
# When calculating score, if not is in front of an adjective then multiply the score of the adjective by -1

def calculate(tokens, lang, dict):
    total = 0
    token_count = len(tokens)
    
    for i in tokens:
        # mult = 1
        # if re.match('^-|^\d', i):
        #     mult = int("".join(re.findall('[^a-z]', i)))
        #     i = i.lstrip(str(mult))
        if i in dict:
            total += dict[i]
            print(i + ":" + str(dict[i]))

    if token_count == 0:
        score = 0
    else:
        score = total / token_count

    return score