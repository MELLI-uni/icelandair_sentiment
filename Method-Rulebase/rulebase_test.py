import Rulebase
import Rulebase_Icelandic
import Update
import pandas as pd

###ENGLISH###
# eng_data1 = '../Data/NLP_English_JAN2022_OPEN.xlsx'
# eng_sheet1 = 'Result 1'
# eng_data2 = '../Data/NPS_freetext_MAY_2022_answers_OPEN.xlsx'
# eng_sheet2 = 'English'

# df_eng1 = Rulebase.init(eng_data1, eng_sheet1)
# df_eng2 = Rulebase.init(eng_data2, eng_sheet2)
# df_eng = Rulebase.combine_df(df_eng1, df_eng2)
# df_eng = Rulebase.separate_multi(df_eng, "EN")

# df_eng_train, df_eng_dev, df_eng_test = Rulebase.train_dev_test_split(df_eng)

# Save for testing purpose
# df_eng_train.to_pickle('./train.pkl')
# df_eng_dev.to_pickle('./dev.pkl')
# df_eng_test.to_pickle('./test.pkl')
#####
# df_eng_train = pd.read_pickle('./train.pkl')
# df_eng_dev = pd.read_pickle('./dev.pkl')
# df_eng_test = pd.read_pickle('./test.pkl')

# df_tune = Rulebase.combine_df(df_eng_train, df_eng_dev)
# df_ttune = Rulebase.combine_df(df_eng_train, df_eng_dev)
######
# Rulebase.train(df_eng_train, "EN")
# Rulebase.train(df_eng_dev, "EN")
# Rulebase.train(df_eng_test, "EN")
# Rulebase.tune_lexicon(df_eng_test, "EN")

#Rulebase.clean_lexicon("EN")
#f1_micro = Rulebase.test_lexicon(df_ttune, "EN")
# f1_micro = 0.9539877300613497

# while f1_micro < 0.97:
# Rulebase.tune_lexicon(df_tune, "EN")
# Rulebase.test_lexicon(df_ttune, "EN")
#Rulebase.test_lexicon(df_eng_test, "EN")


###ICELANDIC###
#isk_data1 = '../Data/NLP_Icelandic_14FEB2022_YTD.xlsx'
#isk_sheet1 = 'Sheet1'
#isk_data2 = '../Data/NPS_freetext_MAY_2022_Icelandic_answers.xlsx'
#isk_sheet2 = 'Icelandic'

#df_isk1 = Rulebase.init(isk_data1, isk_sheet1)
#df_isk2 = Rulebase.init(isk_data2, isk_sheet2)
#df_isk = Rulebase.combine_df(df_isk1, df_isk2)
#df_isk = Rulebase.separate_multi(df_isk, "IS")

#df_isk.to_pickle('./isk_data.pkl')

# df_isk_train, df_isk_dev, df_isk_test = Rulebase.train_dev_test_split(df_isk)

# Save for testing purpose
# df_isk_train.to_pickle('./trainisk.pkl')
# df_isk_dev.to_pickle('./devisk.pkl')
# df_isk_test.to_pickle('./testisk.pkl')

# df_isk_train = pd.read_pickle('./trainisk.pkl')
# df_isk_dev = pd.read_pickle('./devisk.pkl')
# df_isk_test = pd.read_pickle('./testisk.pkl')

# Rulebase.train(df_isk_train, "IS")

df_isk = pd.read_pickle('./isk_data.pkl')
df_train, df_test = Rulebase.train_n_test_split(df_isk)

df_train.to_pickle('./isk_train.pkl')
df_test.to_pickle('./isk_test.pkl')
#print(df_isk)
