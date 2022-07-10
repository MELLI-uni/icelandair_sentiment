import pandas as pd

labeled_english = './eng_total.pkl'
labeled_icelandic = './isk_total.pkl'
unlabeled = './tuning_total.pkl'
unlabeled_english = 'Tuning/tuning_eng.pkl'
unlabeled_icelandic = 'Tuning/tuning_isk.pkl'
unlabeled_other = 'Tuning/tuning_other.pkl'

print("+-----------------------------------+")
print("INFORMATION ON LABELED ENGLISH DATA")
print("+-----------------------------------+")
df_labeled_english = pd.read_pickle(labeled_english)
print("total    ", df_labeled_english.shape[0])
df2 = df_labeled_english.groupby(['Sentiment'])['Sentiment'].count()
print(df2)
print("positive: ")
print("neutral: ")
print("negative: ")
print("+-----------------------------------+")
print("\n")

print("+-------------------------------------+")
print("INFORMATION ON LABELED ICELANDIC DATA")
print("+-------------------------------------+")
df_labeled_icelandic = pd.read_pickle(labeled_icelandic)
print("total: ", df_labeled_icelandic.shape[0])
print("positive: ")
print("neutral: ")
print("negative: ")
print("+-------------------------------------+")
print("\n")

print("+-----------------------------+")
print("INFORMATION ON UNLABELED DATA")
print("+-----------------------------+")
print("total: ", pd.read_pickle(unlabeled).shape[0])
print("english: ", pd.read_pickle(unlabeled_english).shape[0])
print("icelandic: ", pd.read_pickle(unlabeled_icelandic).shape[0])
print("others: ", pd.read_pickle(unlabeled_other).shape[0])
print("+-----------------------------+")
