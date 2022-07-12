import Rulebase_Icelandic
import pandas as pd

df_unlabeled = pd.read_pickle('./tuning_isk.pkl')
cleaned_text = Rulebase_Icelandic.data_cleaning(df_unlabeled)

Rulebase_Icelandic.train_word2vec(cleaned_text)
