import Language
import pandas as pd

tuning_file = '../Data/tuning_total.pkl'
df_tuning = pd.read_pickle(tuning_file)

df_eng, df_isk, df_other = Language.DetectLanguage(df_tuning)

tuning_eng = df_eng.to_pickle('../Data/Tuning/tuning_eng.pkl')
tuning_isk = df_isk.to_pickle('../Data/Tuning/tuning_isk.pkl')
left_overs = df_other.to_pickle('../Data/Tuning/tuning_other.pkl')

print("LANGUAGE DETECTION COMPLETED")
