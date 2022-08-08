import pandas as pd
import Light_Transformer

df_eng = pd.read_pickle('../Data/eng_total.pkl')
del df_eng['id']

df_isk = pd.read_pickle('../Data/isk_total.pkl')
del df_isk['id']

json_eng = df_eng.to_json('eng.json', orient='records', lines=True)
json_isk = df_isk.to_json('isk.json', orient='records', lines=True)

Light_Transformer.test_CNN(json_eng, 'EN')
Light_Transformer.test_CNN(json_isk, 'IS')