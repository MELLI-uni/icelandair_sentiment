

# import nltk
from nltk.corpus import words
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
import string

import spacy
import regex as re

from reynir import Greynir
isk_greynir = Greynir()

# from nltk.metrics.distance import jaccard_distance
# from nltk.util import ngrams
# from nltk.metrics.distance import edit_distance


# USE: spell check
# entries=['spleling', 'mispelling', 'reccomender']

# for entry in entries:
#     temp = [(jaccard_distance(
#             set(ngrams(entry, 2)), 
#             set(ngrams(w, 2))), w) 
#             for w in correct_spellings if w[0]==entry[0]]
#     print(sorted(temp, key = lambda val:val[0])[0][1])

nlp = spacy.load("en_core_web_sm")
suffix_elim = PorterStemmer()
whitelist = list(wn.words()) + words.words()
porter = PorterStemmer()

english_prefixes = {
"anti": "",    # e.g. anti-goverment, anti-racist, anti-war
"auto": "",    # e.g. autobiography, automobile
"de": "",      # e.g. de-classify, decontaminate, demotivate
"dis": "",     # e.g. disagree, displeasure, disqualify
"down": "",    # e.g. downgrade, downhearted
"extra": "",   # e.g. extraordinary, extraterrestrial
"hyper": "",   # e.g. hyperactive, hypertension
"il": "",     # e.g. illegal
"im": "",     # e.g. impossible
"in": "",     # e.g. insecure
"ir": "",     # e.g. irregular
"inter": "",  # e.g. interactive, international
"mega": "",   # e.g. megabyte, mega-deal, megaton
"mid": "",    # e.g. midday, midnight, mid-October
"mis": "",    # e.g. misaligned, mislead, misspelt
"non": "",    # e.g. non-payment, non-smoking
"over": "",  # e.g. overcook, overcharge, overrate
"out": "",    # e.g. outdo, out-perform, outrun
"post": "",   # e.g. post-election, post-warn
"pre": "",    # e.g. prehistoric, pre-war
"pro": "",    # e.g. pro-communist, pro-democracy
"re": "",     # e.g. reconsider, redo, rewrite
"semi": "",   # e.g. semicircle, semi-retired
"sub": "",    # e.g. submarine, sub-Saharan
"super": "",   # e.g. super-hero, supermodel
"tele": "",    # e.g. television, telephathic
"trans": "",   # e.g. transatlantic, transfer
"ultra": "",   # e.g. ultra-compact, ultrasound
"un": "",      # e.g. under-cook, underestimate
"up": "",      # e.g. upgrade, uphill
}

def stem_prefix(word, prefixes, roots):
    original_word = word

    for prefix in sorted(prefixes, key=len, reverse=True):
        word, nsub = re.subn(prefix, "", word)
        if nsub > 0 and word in roots:
            return word
    return original_word

term = stem_prefix("uncomfortable", english_prefixes, whitelist)

def porter_english_plus(word, prefixes=english_prefixes):
    return porter.stem(stem_prefix(word, prefixes, whitelist))

doc = nlp(term)

for token in doc:
    print(token.lemma_)

from spellchecker import SpellChecker

spell = SpellChecker()

word = 'intertainment'
if word not in whitelist:
    print(spell.correction(word))
else:
    print(word)