import Baseline

eng_data = '../Data/NLP_English_JAN2022_OPEN.xlsx'
eng_sheet = 'Result 1'
isk_data = '../Data/NLP_Icelandic_14FEB2022_YTD.xlsx'
isk_sheet = 'Sheet1'

df_eng = Baseline.init(eng_data, eng_sheet, "EN")
df_isk = Baseline.init(isk_data, isk_sheet, "IS")

Baseline.baseline(df_eng, "EN")
Baseline.baseline(df_isk, "IS")