from translate import Translator
import pickle

translator = Translator(to_lang="is")

is_vader_dict = {}

with open('./VADERlex.txt', encoding='utf-8') as f:
    for line in f:
        [key, value, skip1, skip2] = line.split("\t")
        is_word = translator.translate(key)

        is_vader_dict[is_word] = float(value)

is_vader_dict = sorted(is_vader_dict.keys(), key=lambda x:x.lower())

with open('is_vader.pickle', 'wb') as handle:
    pickle.dump(is_vader_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(is_vader_dict)