import Transformer

# eng_data = '../Data/NLP_English_JAN2022_OPEN.xlsx'
# eng_sheet = 'Result 1'
# isk_data = '../Data/NLP_Icelandic_14FEB2022_YTD.xlsx'
# isk_sheet = 'Sheet1'

# df_eng = Transformer.init(eng_data, eng_sheet, "EN")
# df_isk = Transformer.init(isk_data, isk_sheet, "IS")

tuning1 = '../Data/no-sentiment/NPS_freetext_AUG2021_APR2022_answers.xlsx'
sheet1 = 'Result 1'
tuning2 = '../Data/no-sentiment/NPS_freetext_JAN_FEB_2022_answers.xlsx'
sheet2 = 'Result 1'

df_tuning1 = Transformer.init_tuning(tuning1, sheet1)
df_tuning2 = Transformer.init_tuning(tuning2, sheet2)
df_tuning = Transformer.combine_df(df_tuning1, df_tuning2)

df_tuning.to_pickle("./tuning_total.pkl")
