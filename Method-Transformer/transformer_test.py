import Transformer
#mport RobertaTuning
#import IceBERTTuning
#import FileConverter
import pandas as pd

#eng_data1 = './Data/NLP_English_JAN2022_OPEN.xlsx'
#eng_sheet1 = 'Result 1'
#eng_data2 = './Data/NPS_freetext_MAY_2022_answers_OPEN.xlsx'
#eng_sheet2 = 'English'

#df_eng1 = Transformer.init(eng_data1, eng_sheet1, "EN")
#df_eng2 = Transformer.init(eng_data2, eng_sheet2, "EN")
#df_eng = Transformer.combine_df(df_eng1, df_eng2)

#df_eng = pd.read_pickle('../Data/eng_total.pkl')
#tuning_eng = pd.read_pickle('../Data/Tuning/tuning_eng.pkl')

#tuning_eng.to_csv(tuning_file_e, header=None, index=None, sep=' ', mode='w')
#df_eng = Transformer.separate_multi(df_eng, "EN")
#df_eng = Transformer.sentiment_mapping(df_eng)

#Transformer.test_vanilla_basic(df_eng, "EN")
#Transformer.test_vanilla_5fold(df_eng, "EN")
#Transformer.test_tuning_basic(df_eng, "EN")
#Transformer.test_tuning_5fold(df_eng, "EN")
#Transformer.test_xlm_vanilla_5fold(df_eng)

df_isk = pd.read_pickle('../Data/isk_total.pkl')
#tuning_isk = pd.read_pickle('../Data/Tuning/tuning_isk.pkl')
#tuning_isk.to_csv(tuning_file_i, header=None, index=None, sep = ' ', mode='w')

#df_isk = Transformer.separate_multi(df_isk, "IS")
df_isk = Transformer.sentiment_mapping(df_isk)

#Transformer.test_vanilla_basic(df_isk, "IS")
#Transformer.test_vanilla_5fold(df_isk, "IS")
#Transformer.test_tuning_5fold(df_isk, "IS")
#Transformer.test_xlm_vanilla_5fold(df_isk)
Transformer.test_xlm_tuning_5fold(df_isk)

# Convert tuning pickle file into text file
# English tuning file contains 30380 Lines
# Icelandic tuning file contains 8794 Lines

# Tuning Roberta Transformer Model
#df_tuning_eng = pd.read_pickle('/home/jiyoon/work/Data/Tuning/tuning_eng.pkl')
#FileConverter.DFtoTXT(df_tuning_eng, 'tuning_eng.txt')

# English tuning file contains 30380 lines

# Tuning IceBERT Transformer Model
#df_tuning_isk = pd.read_pickle('/home/jiyoon/work/Data/Tuning/tuning_isk.pkl')
#FileConverter.DFtoTXT(df_tuning_isk, 'tuning_isk.txt')

# Icelandic tuning file contains 
