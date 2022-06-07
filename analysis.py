from itertools import islice

import nltk
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

A_INC = 2       # TEMP VALUES
A_DEC = -2      # TEMP VALUES

C_INC = 2       # TEMP VALUES

# list of degree adverbs
# http://en.wiktionary.org/wiki/Category:English_degree_adverbs
ENG_ADV = {'a bit':A_DEC, 'adequately':A_DEC, 'almost':A_DEC, 'barely':A_DEC, 'fairly':A_DEC, 'hardly':A_DEC, 'just enough':A_DEC, 'kind of':A_DEC, 'kinda':A_DEC, 
            'kindof':A_DEC, 'kind-of':A_DEC, 'less':A_DEC, 'little':A_DEC, 'marginal':A_DEC, 'marginally':A_DEC, 'moderately':A_DEC, 'modest':A_DEC, 'nearly':A_DEC, 
            'occasional':A_DEC, 'occasionally':A_DEC, 'partly':A_DEC, 'scarce':A_DEC, 'scarcely':A_DEC, 'seldom':A_DEC, 'slight':A_DEC, 'slightly':A_DEC, 
            'somewhat':A_DEC, 'sort of':A_DEC, 'sorta':A_DEC, 'sortof':A_DEC, 'sort-of':A_DEC, 'sufficiently':A_DEC, 

            '100 percent':A_INC, '100-percent':A_INC, '100%':A_INC, 'a lot':A_INC, 'alot':A_INC, 'absolutely':A_INC, 'amazingly':A_INC, 'awfully':A_INC, 'clearly':A_INC,
            'completely':A_INC, 'considerable':A_INC, 'considerably':A_INC, 'decidedly':A_INC, 'deeply':A_INC, 'enormous':A_INC, 'enormously':A_INC, 'entirely':A_INC, 
            'especially':A_INC, 'exceedingly':A_INC, 'exceptional':A_INC, 'exceptionally':A_INC, 'excessively':A_INC, 'extensively':A_INC, 'extra':A_INC, 'extreme':A_INC, 
            'extremely':A_INC, 'fabulously':A_INC, 'fantastically':A_INC, 'fully':A_INC, 'greatly':A_INC, 'highly':A_INC, 'hugely':A_INC, 'incredible':A_INC, 
            'incredibly':A_INC, 'intensely':A_INC, 'largely':A_INC, 'major':A_INC, 'majorly':A_INC, 'more':A_INC, 'most':A_INC, 'much':A_INC, 'noticeably':A_INC, 
            'particularly':A_INC, 'perfectly':A_INC, 'positively':A_INC, 'pretty':A_INC, 'purely':A_INC, 'quite':A_INC, 'really':A_INC, 'reasonably':A_INC, 
            'remarkably':A_INC, 'so':A_INC, 'strikingly':A_INC, 'strongly':A_INC, 'substantially':A_INC, 'thoroughly':A_INC, 'too':A_INC, 'total':A_INC, 'totally':A_INC, 
            'tremendous':A_INC, 'tremendously':A_INC, 'truly':A_INC, 'uber':A_INC, 'unbelievably':A_INC, 'unusually':A_INC, 'usually':A_INC, 'utter':A_INC, 
            'utterly':A_INC, 'very':A_INC, 'well':A_INC
            }

# http://mylanguages.org/icelandic_adverbs.php
ISK_ADV = {'að hluta':A_DEC, 'af skornum skammti':A_DEC, 'bara nóg':A_DEC, 'fullnægjandi':A_DEC, 'hóflega':A_DEC, 'hóflegur':A_DEC, 'hófsamur':A_DEC, 
            'jaðar':A_DEC, 'lítillega':A_DEC, 'lítilsháttar':A_DEC, 'litla':A_DEC, 'minna':A_DEC, 'nægilega':A_DEC, 'næstum':A_DEC, 'næstum því':A_DEC, 
            'nokkuð':A_DEC, 'örlítið':A_DEC, 'sjaldan':A_DEC, 'stöku sinnum':A_DEC, 'stundum':A_DEC, 'svoleiðis':A_DEC, 'svolítið':A_DEC, 'svona':A_DEC, 
            'varla':A_DEC,
            
            '100 prósent':A_INC, '100-prósent':A_INC, '100%':A_INC, '100 %':A_INC, 'að fullu':A_INC, 'að mestu leyti':A_INC, 'að miklu leyti':A_INC, 
            'að öllu leyti':A_INC, 'aðallega':A_INC, 'afbrigðilegur':A_INC, 'afskaplega':A_INC, 'ákaflega':A_INC, 'ákveðið':A_INC, 'alger':A_INC, 'algerlega':A_INC, 
            'alveg':A_INC, 'auka':A_INC, 'djúpt':A_INC, 'eindregið':A_INC, 'eingöngu':A_INC, 'frábærlega':A_INC, 'frekar':A_INC, 'fullkomlega':A_INC, 'gersamlega':A_INC, 
            'greinilega':A_INC, 'gríðarlega':A_INC, 'jæja':A_INC, 'jákvætt':A_INC, 'líka':A_INC, 'með öllu':A_INC, 'meira':A_INC, 'meiriháttar':A_INC, 'merkilega':A_INC,
            'merkjanlega':A_INC, 'mest':A_INC, 'mikið':A_INC, 'mikill':A_INC, 'mjög':A_INC, 'öfgafullt':A_INC, 'óhóflega':A_INC, 'ótrúlega':A_INC, 'ótrúlegur':A_INC, 
            'óvenju':A_INC, 'óvenjulegur':A_INC, 'rækilega':A_INC, 'raunverulega':A_INC, 'sæmilega':A_INC, 'sætur':A_INC, 'samtals':A_INC, 'sannarlega':A_INC, 
            'sérstaklega':A_INC, 'sláandi':A_INC, 'stórkostlega':A_INC, 'stórkostlegt':A_INC, 'svo':A_INC, 'talsverður':A_INC, 'talsvert':A_INC, 
            'undantekningarlaust':A_INC, 'vel':A_INC, 'venjulega':A_INC, 'verulega':A_INC, 'virkilega':A_INC
            }

# list of negating words
# https://www.wordsense.eu/not/#Icelandic
ENG_NEG = ["aint", "ain't", "arent", "aren't", "cannot", "cant", "can't", "darent", "daren't", "didnt", "didn't", "doesnt", "doesn't", "don't", "dont", "hadnt", "hadn't", "hasnt", 
            "hasn't", "have-not", "havent", "haven't", "isnt", "isn't", "mightnt", "mightn't", "neednt", "needn't", "neither", "never", "none", "nope", "nor", "not", "nothing", 
            "nowhere", "shant", "shan't", "shouldnt", "shouldn't", "wasnt", "wasn't", "wont", "won't", "wouldnt", "wouldn't", 
            ]

ISK_NEG = ["aldrei", "ekkert", "ekki", "enginn", "hvergi", "hvorki", "ne", "neibb", "neitt"
            ]

def negated(input, lang):
    """
    check if there is any negating words in the sentence
    """
    neg_words = []
    if lang == "EN":
        neg_words = ENG_NEG
    elif lang == "IS":
        neg_words = ISK_NEG

    words = [w.lower() for w in input]
    for word in words:
        if word in neg_words:
            return True

    return False

def allcaps(word, score):
    """
    check if there is any allcaps word in the sentence
    """
    if len(word) > 1 and word.isupper():
        score *= C_INC

    return score

def degreed(word, score, lang):
    """
    check if there is a degree adverb in the sentence
    """
    degree_adv = {}
    if lang == "EN":
        degree_adv = ENG_ADV
    elif lang == "IS":
        degree_adv = ISK_ADV
    
    degree = 0
    
    lower_word = word.tolower()
    if lower_word in degree_adv:
        degree = degree_adv[lower_word]

        if score < 0:
            degree *= -1
        
        degree = allcaps(word, degree)

    return degree

class SentimentAnalyzer(object):
    def __init__(self, eng_lexicon="eng_lexicon.txt", isk_lexicon="isk_lexion.txt", emoji_lexicon="emoji_lexicon.txt", emoticon_lexicon="emoticon_lexicon.txt"):
        self.path = "./lexicons/"

        self.eng = self.make_dict(eng_lexicon)
        self.isk = self.make_dict(isk_lexicon)
        self.emo = self.make_emoji_dict(emoji_lexicon, emoticon_lexicon)

    def make_dict(self, lexicon):
        lex_dict = {}

        loc = self.path + lexicon

        with open(loc) as f:
            for line in f:
                [key, value, skip] = line.split("\t")
                lex_dict[key] = float(value)

        return lex_dict

    def make_emoji_dict(self, emoji, emoticon):
        emoji_dict = {}

        emoji_loc = self.path + emoji
        emoticon_loc = self.path + emoticon

        with open(emoji_loc) as f:
            for line in islice(f, 1, None):
                [key, value] = line.split("\t")
                emoji_dict[key] = value

        with open(emoticon_loc) as f:
            for line in f:
                [key, value] = line.split("\t")
                emoji_dict[key] = value

        return emoji_dict

    def convert_to_score(self, input):
        if input:
            print("there is")
        return 0

    @staticmethod
    def exclamation_point(input):
        print("count exclamation point")

    @staticmethod
    def question_mark(input):
        print("question mark")

    @staticmethod
    def count_each(sentiments):
        print("count each")

    def calc_total_score(self, sentiments):
        if sentiments:
            print("exist")
        else:
            total_score = 0
        
        return total_score
