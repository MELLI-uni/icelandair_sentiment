import Baseline
import pandas as pd
import pickle

# eng_data1 = '../Data/NLP_English_JAN2022_OPEN.xlsx'
# eng_sheet1 = 'Result 1'
# eng_data2 = '../Data/NPS_freetext_MAY_2022_answers_OPEN.xlsx'
# eng_sheet2 = 'English'

# df_eng1 = Baseline.init(eng_data1, eng_sheet1, "EN")
# df_eng2 = Baseline.init(eng_data2, eng_sheet2, "EN")
# df_eng = Baseline.combine_df(df_eng1, df_eng2)

# isk_data1 = '../Data/NLP_Icelandic_14FEB2022_YTD.xlsx'
# isk_sheet1 = 'Sheet1'
# isk_data2 = '../Data/NPS_freetext_MAY_2022_Icelandic_answers.xlsx'
# isk_sheet2 = 'Icelandic'

# df_isk1 = Baseline.init(isk_data1, isk_sheet1, "IS")
# df_isk2 = Baseline.init(isk_data2, isk_sheet2, "IS")
# df_isk = Baseline.combine_df(df_isk1, df_isk2)

# Baseline.baseline(df_eng, "EN")
# Baseline.baseline(df_isk, "IS")

df_eng = pd.read_pickle('./baseline_eng.pickle')
df_isk = pd.read_pickle('./baseline_isk.pickle')

Baseline.baseline(df_eng, "EN")
Baseline.baseline(df_isk, "IS")
