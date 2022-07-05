import Transformer
import pandas as pd

#eng_data1 = './Data/NLP_English_JAN2022_OPEN.xlsx'
#eng_sheet1 = 'Result 1'
#eng_data2 = './Data/NPS_freetext_MAY_2022_answers_OPEN.xlsx'
#eng_sheet2 = 'English'

#df_eng1 = Transformer.init(eng_data1, eng_sheet1, "EN")
#df_eng2 = Transformer.init(eng_data2, eng_sheet2, "EN")
#df_eng = Transformer.combine_df(df_eng1, df_eng2)

#Transformer.test_vanilla(df_eng)

df_eng = pd.read_pickle('../Data/eng_total.pkl')
tuning_eng = pd.read_pickle('../Data/Tuning/tuning_eng.pkl')

df_eng = Transformer.sentiment_mapping(df_eng) 
#df_eng = Transformer.data_processing(df_eng, "EN")
print(df_eng)

df_isk = pd.read_pickle('../Data/isk_total.pkl')
tuning_isk = pd.read_pickle('../Data/Tuning/tuning_isk.pkl')

df_isk = Transformer.sentiment_mapping(df_isk)
#df_isk = Transformer.data_processing(df_isk, "IS")

#Transformer.test_vanilla(df_eng, "EN")
#Transformer.test_tuned_basic(df_eng, "EN")
#Transformer.dev_lex(df_eng)
#Transformer.test_tuned(df_eng, "EN")
#Transformer.test_vanilla(df_isk, "IS")
#Transformer.test_tuned_basic(df_isk, "IS")
#Transformer.test_tuned(df_isk, "IS")
