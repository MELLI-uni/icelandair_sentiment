import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.metrics.distance import jaccard_distance
from nltk.corpus import wordnet as wn
from reynir import Greynir
from reynir_correct import check_single
from textblob import TextBlob

import Functions

# list of degree adverbs
# http://en.wiktionary.org/wiki/Category:English_degree_adverbs
ENG_ADV = ['a bit', 'adequately', 'almost', 'barely', 'fairly', 'hardly', 'just enough', 'kind of', 'kinda', 'kindof', 'kind-of', 'less', 'little', 'marginal', 
            'marginally', 'moderately', 'modest', 'nearly', 'occasional', 'occasionally', 'partly', 'scarce', 'scarcely', 'seldom', 'slight', 'slightly', 
            'somewhat', 'sort of', 'sorta', 'sortof', 'sort-of', 'sufficiently', 

            '100 percent', '100-percent', '100%', 'a lot', 'alot', 'absolutely', 'amazingly', 'awfully', 'clearly', 'completely', 'considerable', 'considerably', 
            'decidedly', 'deeply', 'enormous', 'enormously', 'entirely', 'especially', 'exceedingly', 'exceptional', 'exceptionally', 'excessively', 'extensively', 
            'extra', 'extreme', 'extremely', 'fabulously', 'fantastically', 'fully', 'greatly', 'highly', 'hugely', 'incredible', 'incredibly', 'intensely', 'largely', 
            'major', 'majorly', 'more', 'most', 'much', 'noticeably', 'particularly', 'perfectly', 'positively', 'pretty', 'purely', 'quite', 'really', 'reasonably', 
            'remarkably', 'so', 'strikingly', 'strongly', 'substantially', 'thoroughly', 'too', 'total', 'totally', 'tremendous', 'tremendously', 'truly', 'uber', 
            'unbelievably', 'unusually', 'usually', 'utter', 'utterly', 'very', 'well'
            ]

# # http://mylanguages.org/icelandic_adverbs.php
ISK_ADV = ['að hluta', 'af skornum skammti', 'bara nóg', 'fullnægjandi', 'hóflega', 'hóflegur', 'hófsamur', 'jaðar', 'lítillega', 'lítilsháttar', 'litla', 'minna', 
            'nægilega', 'næstum', 'næstum því','nokkuð', 'örlítið', 'sjaldan', 'stöku sinnum', 'stundum', 'svoleiðis', 'svolítið', 'svona', 'varla',
            
            '100 prósent', '100-prósent', '100%', '100 %', 'að fullu', 'að mestu leyti', 'að miklu leyti', 'að öllu leyti', 'aðallega', 'afbrigðilegur', 'afskaplega', 
            'ákaflega', 'ákveðið', 'alger', 'algerlega', 'alveg', 'auka', 'djúpt', 'eindregið', 'eingöngu', 'frábærlega', 'frekar', 'fullkomlega', 'gersamlega', 
            'greinilega', 'gríðarlega', 'jæja', 'jákvætt', 'líka', 'með öllu', 'meira', 'meiriháttar', 'merkilega','merkjanlega', 'mest', 'mikið', 'mikill', 'mjög', 
            'öfgafullt', 'óhóflega', 'ótrúlega', 'ótrúlegur', 'óvenju', 'óvenjulegur', 'rækilega', 'raunverulega', 'sæmilega', 'sætur', 'samtals', 'sannarlega', 
            'sérstaklega', 'sláandi', 'stórkostlega', 'stórkostlegt', 'svo', 'talsverður', 'talsvert', 'undantekningarlaust', 'vel', 'venjulega', 'verulega', 
            'virkilega'
            ]

# list of negating words
# https://www.wordsense.eu/not/#Icelandic
ENG_NEG = ["aint", "ain't", "arent", "aren't", "cannot", "cant", "can't", "darent", "daren't", "didnt", "didn't", "doesnt", "doesn't", "don't", "dont", "hadnt", "hadn't", "hasnt", 
            "hasn't", "have-not", "havent", "haven't", "isnt", "isn't", "mightnt", "mightn't", "neednt", "needn't", "neither", "never", "none", "nope", "nor", "not", "nothing", 
            "nowhere", "shant", "shan't", "shouldnt", "shouldn't", "wasnt", "wasn't", "wont", "won't", "wouldnt", "wouldn't", 
            ]

ISK_NEG = ["aldrei", "ekkert", "ekki", "enginn", "hvergi", "hvorki", "ne", "neibb", "neitt"
            ]

ENG_VOCAB = wn.all_lemma_names()
ISK_VOCAB = []


# check if the word is an actual word
# https://www.geeksforgeeks.org/correcting-words-using-nltk-in-python/
def ENG_spell_check(word):
    if wn.synsets(word):
        word = word
    else:
        temp = [(jaccard_distance(set(ngrams(word,2)),
                                    set(ngrams(w, 2))), w)
                for w in ENG_VOCAB if w[0] == word[0]]
        word = sorted(temp, key=lambda val:val[0])[0][1]
    
    return word

# https://github.com/mideind/GreynirCorrect
def ISK_spell_check(word):
    word = check_single(word)
    word = word.tidy_text
    
    return word

def update_lexicons(df):
    eng_lexicon = "./lexicons/eng_lexicon.txt"
    #isk_lexicon = "./lexicons/isk_lexicon.txt"

    eng_dict = {}
    #isk_dict = {}

    f = open(eng_lexicon, 'r+', encoding='utf-8')

    for lines in f:
        [key, skip, scores] = lines.split('\t')
        eng_dict[key] = [float(x) for x in scores.strip('\n[]').split(', ')]

    f.seek(0)

    for row in df.itertuples(index=False):
        new_word = row.answer_freetext_value
        new_score = row.Sentiment

        if new_word in eng_dict:
            eng_dict[new_word].append(new_score)
        else:
            eng_dict[new_word] = [new_score]

    for key, value in eng_dict.items():
        f.write(key + "\t" + str(sum(value) / len(value)) + "\t" + str(value) + "\n")

    f.close()

# how to use greynir
# text
# g = Greynir()
# job = g.submit(text)

# iterate through sentences and parse each one
# for sent in job:
#   sent.parse()
#   #sent.lemmas