import Rulebase

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

# Rulebase.update_lexicon(df_eng_train)
# Rulebase.tune_lexicon(df_eng_dev)
# Rulebase.test_lexicon(df_eng_test)


###ICELANDIC###
# isk_data1 = '../Data/NLP_Icelandic_14FEB2022_YTD.xlsx'
# isk_sheet1 = 'Sheet1'
# isk_data2 = '../Data/NPS_freetext_MAY_2022_Icelandic_answers.xlsx'
# isk_sheet2 = 'Icelandic'

# df_isk1 = Rulebase.init(isk_data1, isk_sheet1)
# df_isk2 = Rulebase.init(isk_data2, isk_sheet2)
# df_isk = Rulebase.combine_df(df_isk1, df_isk2)
# df_isk = Rulebase.separate_multi(df_isk, "IS")

# df_isk_train, df_isk_dev, df_isk_test = Rulebase.train_dev_test_split(df_isk)

# Rulebase.update_lexicon(df_isk_train)
# Rulebase.tune_lexicon(df_isk_dev)
# Rulebase.test_lexicon(df_isk_test)


### GENERAL TESTING ###
#text = "It would be better if your baggage allowance was clear on the ticket rather than having to go back in to check on the web via the ticket type."
#text = "Very uncomfortable seats. Horrible experience"
#text = "I did not like the flight because the seats were uncomfortable"
#text = "This was one of the nicest, efficient, and pleasant flight crew we ever have experienced."
#text = "very quiet flight attendants letting passengers relax"
text = "intertainment"
Rulebase.process_eng(text, 1)
#Rulebase.sample_isk()