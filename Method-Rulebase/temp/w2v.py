#import Rulebase_Icelandic
#import pandas as pd

#df_unlabeled = pd.read_pickle('./tuning_isk.pkl')
#cleaned_text = Rulebase_Icelandic.data_cleaning(df_unlabeled)

#Rulebase_Icelandic.train_word2vec(cleaned_text

from gensim.models import KeyedVectors

model = KeyedVectors.load('w2v_lem_optMSL_s350_w1_a002_e20_n13_t00001_exp04_mint5.kv')
vector = model['gott']

print(vector.shape)
