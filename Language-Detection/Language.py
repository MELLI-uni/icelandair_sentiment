import cld3
import pandas as pd

def clean(df):
    header = list(df.columns)

    to_leave = ['answer_freetext_value']
    
    for h in header:
        if h not in to_leave:
            del df[h]

    print("***DATA CLEANING***")
    print("Before Cleaning: ", df.shape)

    df.dropna(inplace=True)

    df_temp = df.loc[(df['answer_freetext_value'].str.split().str.len()) < 2]
    df.drop(df_temp.index, inplace=True)
    
    df.reset_index(inplace=True, drop=True)

    print("After Cleaning: ", df.shape)
    print("-------------------\n")

    return df

def label(df):
    df['Language'] = df['answer_freetext_value'].apply(lambda x: cld3.get_language(x))
    df.dropna(subset=['Language'], inplace=True)
    df['Language'] = df['Language'].apply(lambda x: x.language)

    return df

def DetectLanguage(df):
    df = clean(df)
    df = label(df)

    df_eng = df[df['Language'].isin(['en'])]
    del df_eng['Language']

    df_isk = df[df['Language'].isin(['is'])]
    del df_isk['Language']

    df_other = df[~df['Language'].isin(['en', 'is'])]
    del df_other['Language']

    print("***RESULT***")
    print("English Data: ", df_eng.shape)
    print("Icelandic Data: ", df_isk.shape)
    print("Other Language: ", df_other.shape)
    print("------------")

    return df_eng, df_isk, df_other

