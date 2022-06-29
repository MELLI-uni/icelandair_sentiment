import xlwings as xws
import string
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from tabulate import tabulate

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

    return df

def init_tuning(file_name, sheet_name):
    wb = xws.Book(file_name)
    sheet = wb.sheets[sheet_name].used_range

    df = sheet.options(pd.DataFrame, index=False, header=True).value

    header = list(df.columns)

    # Leave only id, freetext, Sentiment column
    to_leave = ['id', 'answer_freetext_value']

    for h in header:
        if h not in to_leave:
            del df[h]

    # Drop all rows with either empty freetext or empty sentiment
    df.dropna(subset = ['answer_freetext_value'], inplace=True)

    return df
    
def combine_df (df1, df2):
    df = pd.concat([df1, df2], ignore_index=True)

    return df


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

    return [precision, recall, f1_gen], [f1_micro, f1_macro]

def display(scores, f1s):
    """
    display function shows the results in a table

    : param scores: 3 by 3 np array with Sentiments as columns and [Precision, Recall, F1] as indexes
    : param f1s: list containing f1 microaverage and f1 macroaverage
    """
    # Compile scores in panda dataframe
    score_compile = np.array([scores[0], scores[1], scores[2]])
    f1_average = np.array([f1s[0], f1s[1]])
    df_score = pd.DataFrame(data=score_compile, index=['Precision', 'Recall', 'F1'], columns=CATEGORIES)
    df_average = pd.DataFrame(data=f1_average, index=['F1 Microaverage', 'F1 Macroaverage'], columns=['Scores'])

    # Print dataframe in tabular format
    print(tabulate(df_score, headers='keys', tablefmt='pretty'))
    print(tabulate(df_average, headers='keys', tablefmt='pretty'))