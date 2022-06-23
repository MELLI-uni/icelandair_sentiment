import regex as re
from nltk.corpus import words
from nltk.corpus import wordnet as wn

english_words = list(wn.words()) + words.words() + ['covid', 'COVID', 'corona', 'Icelandair', 'IcelandAir', 'icelandair', 'icelandAir', 'Iceland air', 'Iceland Air', 'iceland air', 'iceland Air']

# ENG_PREFIX = {'anti':'against ', 'de':'opposite ', 'dis':'not ', 'en':'cause to ', 'em':'cause to ', 'fore':'before ', 'in':'not ', 'im':'not ', 'inter':'between ',
#                 'mid':'middle ', 'mis':'wrongly ', 'non':'not ', 'over':'over ', 'pre':'before ', 're':'again ', 'semi':'half ', 'sub':'under ', 'super':'above ',
#                 'trans':'across ', 'un':'not ', 'under':'under '
#             }

# # ENG_SUFFIX = {'able':'can be done', 'ible':'can be done', 'al':'having characteristics of', 'ial': 'having characteristics of ', 'en': 'made of', 'ful':'full of',
# #                 'ic':'having characteristics of', 'ion':'process', 'tion':'process', 'ation':'process', 'ition':'process', 'ity':'state of', 'ty':'state of', 
# #                 'ive':'adjective form of a noun', 'ative':'adjective form of a noun', 'itive':'adjective form of a noun', 'less':'without', 'ly':'characteristics of',
# #                 'ment':'process', 'ness':'state of', 'ous':'possessing quality of', 'eous':'possessing quality of', 'ious':'possessing quality of', 'y':'characterized by'
# #             }

ENG_SUFFIX = {'able':'', 'ible':'', 'al':'', 'ial':'', 'ed':'', 'en':'', 'er':'', 'ful':'full of ',
                'ic':'', 'ing':'', 'ion':'', 'tion':'', 'ation':'', 'ition':'', 'ity':'', 'ty':'', 
                'ive':'', 'ative':'', 'itive':'', 'less':'without ', 'ly':'',
                'ment':'', 'ness':'', 'ous':'', 'eous':'', 'ious':'', 'y':''
            }

# def stem_prefix(word, prefixes, roots):
#     original_word = word

#     for prefix in sorted(prefixes, key=len, reverse=True):
#         if original_word.startswith(prefix):
#             word, nsub = re.subn(prefix, "", original_word)
#             if nsub > 0 and word in roots:
#                 return prefixes[prefix] + word
    
#     return None

def stem_suffix(word, suffixes, roots):
    original_word = word

    for suffix in sorted(suffixes, key=len, reverse=True):
        if original_word.endswith(suffix):
            word, nsub = re.subn(suffix, "", original_word)
            if word in roots:
                return suffixes[suffix] + word
            elif word+'e' in roots:
                return suffixes[suffix] + word+'e'
            elif word[-1] == word[-2:-1] and word[:-1] in roots:
                return suffixes[suffix] + word[:-1]
    
    return None

# stem_prefix_check = ['uncomfortable', 'disagree', 'misunderstand', 'impossible', 'agree']
stem_suffix_check = ['cancellation']

# # for word in stem_prefix_check:
# #     print(word, stem_prefix(word, ENG_PREFIX, english_words))

for word in stem_suffix_check:
    print(word, stem_suffix(word, ENG_SUFFIX, english_words))

if 'pax' in english_words:
    print(True)