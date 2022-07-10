import pandas as pd

labeled_english = './eng_total.pkl'
labeled_icelandic = './isk_total.pkl'
unlabled = './tuning_total.pkl'
unlabeled_english = 'Tuning/tuning_eng.pkl'
unlabeled_icelandic = 'Tuning/tuning_isk.pkl'
unlabeled_other = 'Tuning/tuning_other.pkl'

print("+-----------------------------------+")
print("|INFORMATION ON LABELED ENGLISH DATA|")
print("+-----------------------------------+")
df_labeled_english = pd.read_pickle(labeled_english)
print("|total: ", df_labeled_english.shape[0])
print("|positive: ")
print("|neutral: ")
print("|negative: ")
print("\n\n")

print("INFORMATION ON LABELED ICELANDIC DATA")
print("-------------------------------------")
print("total: ")
print("positive: ")
print("negative: ")
print("\n\n")

print("INFORMATION ON UNLABELED DATA")
print("-----------------------------")
print("total: ")
print("english: ")
print("icelandic: ")
print("others: ")
print("\n\n")
