import Transformer

### ENGLISH ###
eng_data1 = '../Data/NLP_English_JAN2022_OPEN.xlsx'
eng_sheet1 = 'Result 1'
eng_data2 = '../Data/NPS_freetext_MAY_2022_answers_OPEN.xlsx'
eng_sheet2 = 'English'

df_eng1 = Transformer.init(eng_data1, eng_sheet1, "EN")
df_eng2 = Transformer.init(eng_data2, eng_sheet2, "EN")
df_eng = Transformer.combine_df(df_eng1, df_eng2)


### ICELANDIC ###
isk_data1 = '../Data/NLP_Icelandic_14FEB2022_YTD.xlsx'
isk_sheet1 = 'Sheet1'
isk_data2 = '../Data/NPS_freetext_MAY_2022_Icelandic_answers.xlsx'
isk_sheet2 = 'Icelandic'

df_isk1 = Transformer.init(isk_data1, isk_sheet1, "IS")
df_isk2 = Transformer.init(isk_data2, isk_sheet2, "IS")
df_isk = Transformer.combine_df(df_isk1, df_isk2)