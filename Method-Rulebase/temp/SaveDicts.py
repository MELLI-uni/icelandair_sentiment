import pickle
from itertools import islice
from translate import Translator

translator = Translator(to_lang="is")

dictionaries  = []

# Make English emoji dictionary from emoticon and emoji file
# Make Icelandic emoji dictionary by translating English into Icelandic (with translate api)
emoji_dict_eng = {}
emoji_dict_isk = {}
with open('../../lexicons/emoji.txt', encoding='utf-8') as f:
    for line in islice(f, 1, None):
        [key, value] = line.split("\t")
        emoji_dict_eng[key] = value.strip()
        emoji_dict_isk[key] = translator.translate(value.strip())
with open('../../lexicons/emoticon.txt', encoding='utf-8') as f:
    for line in f:
        [key, value] = line.split("\t")
        emoji_dict_eng[key] = value.strip()
        emoji_dict_isk[key] = translator.translate(value.strip())
dictionaries.append(emoji_dict_eng)
dictionaries.append(emoji_dict_isk)

# Make flight and destination list from destination lexicon
flight_list = []
with open('../../lexicons/destination.txt', encoding='utf-8') as f:
    for line in f:
        flight_list.append(line.strip())

# Make icelandic stopword list from stopword lexicon
isk_stop = []
with open('../../lexicons/isk_stop.txt', encoding='utf-8') as f:
    for line in f:
        isk_stop.append(line.strip())

# Store data (serialize)
with open('dictionaries.pickle', 'wb') as handle:
    pickle.dump(dictionaries, handle, protocol=pickle.HIGHEST_PROTOCOL)