"""
File to convert western emoticon into a text file

Emoticon list from:
    https://en.wikipedia.org/wiki/List_of_emoticons
    Section: Western, Sideways Latin-only emoticons
    
Requirements:
    File_name should be setted to "western-emoticons.xlsx"
    All data must be placed on "Sheet1"
    Cells of excel MUST be formatted to 'text' when copying and pasting
"""

import pandas as pd
import re

file_name = "western-emoticons.xlsx"
sheet = "Sheet1"
df = pd.read_excel(io = file_name, sheet_name = sheet)

# Eliminate all reference mark in format [#] and more than one spaces
# Copies preceding row for meaning if empty
df_new = df["Meaning"].str.replace(r'(\[[0-9]+\])+', '', regex=True).str.replace(r'\s+', ' ', regex=True).ffill()
del df["Meaning"]

# Converts every row of emoticons to one list and remove all 'nan' from the list
df["Emoticon"] = df.values.tolist()
df["Emoticon"] = df["Emoticon"].apply(lambda x: [i for i in x if (pd.isnull(i)) == False])

# Append "Meaning" column to the end and drop all columns except "Emoticon" and "Meaning"
df["Meaning"] = df_new
df.drop(df.columns.difference(['Emoticon', 'Meaning']), axis=1, inplace=True)

# Explode the "Emoticon" column
df = df.explode('Emoticon')

df.to_csv('./lexicons/emoticon-lexicon.txt', header=None, index=None, sep='\t', mode='w')