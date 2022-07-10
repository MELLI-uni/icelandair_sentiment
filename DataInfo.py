import pandas as pd

labeled_english = './eng_total.pkl'
labeled_icelandic = './isk_total.pkl'
unlabeled = './tuning_total.pkl'
unlabeled_english = 'Tuning/tuning_eng.pkl'
unlabeled_icelandic = 'Tuning/tuning_isk.pkl'
unlabeled_other = 'Tuning/tuning_other.pkl'

print("INFORMATION ON LABELED ENGLISH DATA")
df_labeled_english = pd.read_pickle(labeled_english)
df_labeled_english2 = df_labeled_english.groupby(['Sentiment'])['Sentiment'].count()
print("total", "\t\t", df_labeled_english.shape[0])
print(df_labeled_english2)
print("\n")

print("INFORMATION ON LABELED ICELANDIC DATA")
df_labeled_icelandic = pd.read_pickle(labeled_icelandic)
df_labeled_icelandic2 = df_labeled_icelandic.groupby(['Sentiment'])['Sentiment'].count()
print("total", "\t\t", df_labeled_icelandic.shape[0])
print(df_labeled_icelandic2)
print("\n")

print("INFORMATION ON UNLABELED DATA")
print("total", "\t\t", pd.read_pickle(unlabeled).shape[0])
print("english", "\t\t", pd.read_pickle(unlabeled_english).shape[0])
print("icelandic", "\t\t", pd.read_pickle(unlabeled_icelandic).shape[0])
print("others", "\t\t", pd.read_pickle(unlabeled_other).shape[0])
