from zlib import Z_NO_COMPRESSION
import regex as re
import xlwings as xws
import string

from itertools import groupby
from itertools import islice
from collections import Counter

import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tabulate import tabulate

import gensim
from gensim.parsing.preprocessing import STOPWORDS      # List of English Stopwords

import spacy
from spacy.lang.en import English
from nltk.corpus import words
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

import gensim.downloader

from reynir import Greynir
from reynir_correct import check_single
from reynir_correct import check
from google_trans_new import google_translator

eng_spacy = spacy.load('en_core_web_sm')
spell = SpellChecker()
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
isk_greynir = Greynir()
translator = google_translator()       # Used to translate emoji and emoticon description to icelandic (lang code: 'is')

emoji_dict = {}
with open('../lexicons/emoji.txt', encoding='utf-8') as f:
    for line in islice(f, 1, None):
        [key, value] = line.split("\t")
        emoji_dict[key] = value.strip()
with open('../lexicons/emoticon.txt', encoding='utf-8') as f:
    for line in f:
        [key, value] = line.split("\t")
        emoji_dict[key] = value.strip()

flight_list = []
with open('../lexicons/destination.txt', encoding='utf-8') as f:
    for line in f:
        flight_list.append(line.strip())

isk_stop = []
with open('../lexicons/isk_stop.txt', encoding='utf-8') as f:
    for line in f:
        isk_stop.append(line.strip())

english_words = list(wn.words()) + words.words() + ['covid', 'COVID', 'corona', 'Icelandair', 'IcelandAir', 'icelandair', 'icelandAir', 'Iceland air', 'Iceland Air', 'iceland air', 'iceland Air']

eng_negating = r'\s[Bb]ut\.*,*\s|\s[Hh]owever\.*,*\s'
isk_negating = r'\s[Ee]n\.*,*\s|\s[Nn]ema\.*,*\s'

A_INC = 1
A_DEC = -0.5

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

# Reference: https://www.scholastic.com/content/dam/teachers/lesson-plans/migrated-files-in-body/prefixes_suffixes.pdf
# ENG_PREFIX = {'anti':'against', 'de':'opposite', 'dis':'not', 'en':'cause to', 'em':'cause to', 'fore':'before', 'in':'not', 'im':'not', 'inter':'between',
#                 'mid':'middle', 'mis':'wrongly', 'non':'not', 'over':'over', 'pre':'before', 're':'again', 'semi':'half', 'sub':'under', 'super':'above',
#                 'trans':'across', 'un':'not', 'under':'under'
#             }

ENG_PREFIX = {'dis':'not', 'in':'not', 'im':'not', 'il':'not', 'ir':'not', 'non':'not', 'un':'not'
            }

# ISK_PREFIX = {

#             }

# ENG_SUFFIX = {'able':'can be done', 'ible':'can be done', 'al':'having characteristics of', 'ial': 'having characteristics of ', 'en': 'made of', 'ful':'full of',
#                 'ic':'having characteristics of', 'ion':'process', 'tion':'process', 'ation':'process', 'ition':'process', 'ity':'state of', 'ty':'state of', 
#                 'ive':'adjective form of a noun', 'ative':'adjective form of a noun', 'itive':'adjective form of a noun', 'less':'without', 'ly':'characteristics of',
#                 'ment':'process', 'ness':'state of', 'ous':'possessing quality of', 'eous':'possessing quality of', 'ious':'possessing quality of', 'y':'characterized by'
#             }

ENG_SUFFIX = {'able':'', 'ible':'', 'al':'', 'ial':'', 'ed':'', 'en':'', 'er':'', 'ful':'', 'fully':'',
                'ic':'', 'ing':'', 'ion':'', 'tion':'', 'ation':'', 'ition':'', 'ity':'', 'ty':'', 
                'ive':'', 'ative':'', 'itive':'', 'ly':'',
                'ment':'', 'ness':'', 'ous':'', 'eous':'', 'ious':'', 'y':''
            }

# ISK_SUFFIX = {

#             }

#ISK_NEG = ["ekki"]
ISK_NEG = ["aldrei", "ekkert", "ekki", "enginn", "hvergi", "hvorki", "ne", "neibb", "neitt"]
ENG_NEG = ["neither", "never", "none", "nope", "nor", "not", "nothing", "nowhere"]
NEUTRAL_SKIP = ["N/A", "n/a", "na", "N/a", "n/A", "NA"]

def init(file_name, sheet_name):
    """
    init function launches a password-protected excel file for the user to open and changes it into a dataframe
    all other columns exclusing 'id', 'answer_freetext_value', 'sentiment' is eliminated from the dataframe
    responses in 'answer_freetext_value' is lemmatized
    'sentiment' column is separated into three boolean columns: 'Positive', 'Negative', 'Neutral'

    : param file_name: location/name of the excel file to open
    : param sheet_name: name of the sheet to open

    : return: data in pandas dataframe
    """

    wb = xws.Book(file_name)
    sheet = wb.sheets[sheet_name].used_range

    df = sheet.options(pd.DataFrame, index=False, header=True).value

    header = list(df.columns)

    # Leave only id, freetext, Sentiment column
    to_leave = ['id', 'answer_freetext_value', 'Sentiment']

    for h in header:
        if h not in to_leave:
            del df[h]

    # Lowercase & Strip trailing blanks for sentiment
    df['Sentiment'] = df['Sentiment'].str.lower().str.strip()

    # Drop all rows with either empty freetext or empty sentiment
    df.dropna(subset = ['answer_freetext_value'], inplace=True)
    df.dropna(subset = ['Sentiment'], inplace=True)

    return df

def combine_df(df1, df2):
    """
    combine_df function concatenates two dataframe into one larger dataframe

    : param df1, df2: dataframes to be concatenated

    : return: concatenated dataframe
    """
    df = pd.concat([df1, df2], ignore_index=True)

    return df

def separate_multi(df, lang):
    """
    separate_multi function separates rows of dataframe with multiple sentiments
    Separation Rule:
        1. if there is a new line, separate at new line
        2. if there is a negating conjugation, separate at negating conjugation
        3. if there is a period, separate at period
        4. if there is a comma, separate at comma
        else. delete row from dataframe
    """

    df['Sentiment'] = list(df['Sentiment'].str.split(r'\W+'))

    df_multi = df.loc[(df['Sentiment']).str.len() > 1]
    df.drop(df_multi.index, inplace=True)
    df = df.explode(['Sentiment'])

    df_multi['ct_senti'] = (df_multi['Sentiment']).str.len()

    df_multi['ct_sep'] = (df_multi['answer_freetext_value']).str.split(r'\n+').str.len()
    df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep']), 'answer_freetext_value'] = (df_multi['answer_freetext_value']).str.split(r'\n+')
    df_sep = df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep'])]
    df_multi.drop(df_sep.index, inplace=True)
    del df_multi['ct_sep']

    if lang == 'EN':
        negating_pattern = eng_negating
    elif lang == 'IS':
        negating_pattern = isk_negating
    
    df_multi['ct_sep'] = (df_multi['answer_freetext_value']).str.split(negating_pattern).str.len()
    df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep']), 'answer_freetext_value'] = (df_multi['answer_freetext_value']).str.split(negating_pattern)
    df_temp = df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep'])]
    df_sep = pd.concat([df_sep, df_temp], sort=False)
    df_multi.drop(df_temp.index, inplace=True)
    del df_multi['ct_sep']

    df_multi['ct_sep'] = (df_multi['answer_freetext_value']).str.strip(r'[.!?]').str.split(r'[.!?]').str.len()
    df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep']), 'answer_freetext_value'] = (df_multi['answer_freetext_value']).str.strip(r'[.!?]').str.split(r'[.!?]')
    df_temp = df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep'])]
    df_sep = pd.concat([df_sep, df_temp], sort=False)
    df_multi.drop(df_temp.index, inplace=True)
    del df_multi['ct_sep']

    df_multi['ct_sep'] = (df_multi['answer_freetext_value']).str.split(r'[,;]').str.len()
    df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep']), 'answer_freetext_value'] = (df_multi['answer_freetext_value']).str.split(r'[,;]')
    df_temp = df_multi.loc[(df_multi['ct_senti'] == df_multi['ct_sep'])]
    df_sep = pd.concat([df_sep, df_temp], sort=False)
    df_multi.drop(df_temp.index, inplace=True)
    del df_multi['ct_sep']
    
    del df_sep['ct_senti']
    del df_sep['ct_sep']
    df_sep = df_sep.explode(['Sentiment', 'answer_freetext_value'])

    df = pd.concat([df, df_sep], ignore_index=True, sort=False)

    return df

def sentiment_to_val(df):
    df.loc[df['Sentiment'] == 'positive', 'Sentiment'] = 1
    df.loc[df['Sentiment'] == 'negative', 'Sentiment'] = -1
    df.loc[df['Sentiment'] == 'neutral', 'Sentiment'] = 0

    return df

def train_dev_test_split(df):
    """
    train_dev_test_split function divides the dataframe into training set, development set, and testing set
    composition ratio: 70% training, 20% dev, 10% testing

    : param df: pandas dataframe that would be separated into three sets

    : return: pandas dataframe divided into training, dev, and testing
    """

    df_train, df_others = train_test_split(df, test_size=0.3, shuffle=True)
    df_dev, df_test = train_test_split(df_others, test_size=0.33, shuffle=True)

    return df_train, df_dev, df_test

### TO BE DELETED
def train_n_test_split(df):
    """
    train_dev_test_split function divides the dataframe into training set, development set, and testing set
    composition ratio: 70% training, 20% dev, 10% testing

    : param df: pandas dataframe that would be separated into three sets

    : return: pandas dataframe divided into training, dev, and testing
    """

    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)

    return df_train, df_test

def stem_prefix(word, prefixes, roots):
    original_word = word

    for prefix in sorted(prefixes, key=len, reverse=True):
        if original_word.startswith(prefix):
            word, nsub = re.subn(prefix, "", original_word)
            if nsub > 0 and word in roots:
                return prefixes[prefix], word
    
    return None

def stem_suffix(word, suffixes, roots):
    original_word = word

    for suffix in sorted(suffixes, key=len, reverse=True):
        if original_word.endswith(suffix):
            word, nsub = re.subn(suffix, "", word)
            if word in roots:
                return suffixes[suffix], word
            elif word+'e' in roots:
                return suffixes[suffix], word+'e'
            elif word[-1] == word[-2:-1] and word[:-1] in roots:
                return suffixes[suffix], word[:-1]

    return None

def spell_correct_eng(input, lexicon):
    tokens = word_tokenize(input)

    corrected_tokens = []

    for token in tokens:
        if token.lower() not in english_words:
            # temp = spell.correction(token)

            # if temp in english_words:
            #     corrected_tokens.append(temp)
            # continue
            # corrected_tokens.append(spell.correction(token))
            token = spell.correction(token)

        temp = stem_prefix(token, ENG_PREFIX, lexicon)
        if temp != None:
            corrected_tokens.append(temp[0] + " " + temp[1])
        else: 
            corrected_tokens.append(token)

    return " ".join(t for t in corrected_tokens)

def lemmatize_isk(input):
    token_list = []
    job = isk_greynir.submit(input)

    for sent in job:
        sent.parse()
        if sent.tree is None:
            for t in sent.tokens:
                token_list.append(t.txt)
        else:
            token_list.extend(sent.lemmas)

    return " ".join([token for token in token_list if token not in string.punctuation])

def weight_tokens(unique_tag_list, tag_list, mark_list):
    tag_dict = {}

    weight = 1
    if "NOUN" in unique_tag_list:
        tag_dict["NOUN"] = 0.5

    if "VERB" in unique_tag_list:
        tag_dict["VERB"] = weight
        weight += 1

    if "ADV" in unique_tag_list:
        tag_dict["ADV"] = weight
        weight += 1

    if "ADJ" in unique_tag_list:
        tag_dict["ADJ"] = weight
        weight += 1

    apply_weight = map(lambda tag: tag_dict[tag], tag_list)
    weight_list = list(apply_weight)
    weight_list = np.add(weight_list, mark_list)

    return weight_list

def process_eng(input, sentiment):
    doc = eng_spacy(input)

    text = []
    tag = []
    polarity_pure = []
    polarity = []

    token_score = sentiment
    auxiliary = []
    mark_down = 0
    mark_list = []
    negate = False
    comp_neg = False

    score_deg = 0
    flag_deg = False

    for token in doc:
        if token.text in flight_list:
            continue

        if (
            token.text == 'have' or
            token.text == 'has' or
            token.text == 'had'
        ):
            continue

        if token.pos_ == 'AUX':
            auxiliary.append(True)
            continue

        if len(auxiliary) == 1:
            auxiliary = []

        if token.lemma_ in ENG_ADV and token.pos_ == 'ADV':
            score_deg += ENG_ADV[token.lemma_]
            flag_deg = True

            if token == doc[-1]:
                polarity[-1] += score_deg
                score_deg = 0
                flag_deg = False

            continue

        if len(auxiliary) >= 2:
            if(token.pos_ == 'ADJ'):
                auxiliary = []
                text.append(token.lemma_)
                tag.append(token.pos_)
                mark_list.append(mark_down)
                if flag_deg == True:
                    polarity.append(-token_score + score_deg)
                    score_deg = 0
                    flag_deg = False
                else:
                    polarity.append(-token_score)
                polarity_pure.append(-token_score)
                comp_neg = True
            continue

        # TODO: double check this portion
        if token.dep_ == 'cc':
            comp_neg = False
            continue

        if token.dep_ == 'prep':
            continue

        if token.dep_ == 'neg':
            negate = True
            continue

        if token.dep_ == 'mark':
            mark_down -= 0.25
            if comp_neg == True:
                token_score = -token_score
            continue

        if token.pos_ == r'[\.\?!]':
            token_score = sentiment
            continue

        if (
            token.pos_ == 'ADP' or 
            token.pos_ == 'CONJ' or
            token.pos_ == 'CCONJ' or
            token.pos_ == 'DET' or
            token.pos_ == 'INTJ' or
            token.pos_ == 'NUM' or
            token.pos_ == 'PART' or
            token.pos_ == 'PRON' or
            token.pos_ == 'PROPN' or
            token.pos_ == 'PUNCT' or
            token.pos_ == 'SCONJ' or
            token.pos_ == 'SYM' or
            token.pos_ == 'X' or
            token.pos_ == 'SPACE'):
            continue

        if negate == True:
            text.append(token.lemma_)
            tag.append(token.pos_)
            mark_list.append(mark_down)
            if flag_deg == True:
                polarity.append(-token_score + score_deg)
                score_deg = 0
                flag_deg = False
            else:
                polarity.append(-token_score)
            polarity_pure.append(-token_score)
            negate = False
            continue

        if token.pos_ == 'NOUN' and token.lemma_ in text:
            continue

        if token.lemma_ in STOPWORDS:
            continue

        text.append(token.lemma_)
        mark_list.append(mark_down)
        if flag_deg == True:
            polarity.append(token_score + score_deg)
            flag_deg = False
        else:
            polarity.append(token_score)
        polarity_pure.append(token_score)
        tag.append(token.pos_)

    list_tag = Counter(tag).keys()

    weight = weight_tokens(list_tag, tag, mark_list)

    return [text, polarity_pure, polarity, weight]

def process_isk(input, sentiment):
    text = []
    polarity_pure = []
    polarity = []
    weight = []

    job = isk_greynir.submit(input)

    for sent in job:
        sent.parse()
        if sent.tree is None:
            for t in sent.tokens:
                if t.txt.lower() not in isk_stop:
                    text.append(t.txt.lower())
                    polarity_pure.append(sentiment)
        else:
            for l in sent.lemmas:
                if l.lower() not in isk_stop:
                    text.append(l.lower())
                    polarity_pure.append(sentiment)

    return [text, polarity_pure, polarity, weight]

def temp(input, sentiment):
    text = []
    tag = []
    polarity_pure = []
    polarity = []
    weight = []

    token_score = sentiment
    auxiliary = []
    mark_down = 0
    mark_list = []
    negate = False
    comp_neg = False

    score_deg = 0
    flag_deg = False

    sentence_tree = []

    prev = ""

    input = list(check(input))

    for pg in input:
        for sent in pg:
            sent.parse()
            if sent.tree is None:
                print("No Parse Available")
                for t in sent.tokens:
                    text.append(t.txt)
            else:
                text.extend([l for l in sent.lemmas if l not in string.punctuation])
                sentence_tree.extend(sent.tree.view.split("\n"))

    # for branch in sentence_tree:
    #     print(branch)

    label_list = []
    ct = 0

    for branch in sentence_tree:
        if branch == "S0":
            continue
        line, level = re.subn(r'\s\s', '', branch)

        if re.search(r": ", line):
            prev_tags = ",".join(label_list)
            tag, skip = line.split(":")
            print(prev_tags, ",", tag.replace("+-", ""), " ", text[ct])
            ct += 1
            continue

        if level >= len(label_list):
            label_list.append(line.replace("+-", ""))

        print(level, " ", line)
    
    print(text)
        # line, num = re.subn(r'\+-', "", branch)
        # result, level = re.subn(r'\s\s', "", line)
        # print(result, level)

    # for branch in sentence_tree:
    #     if re.search(r": '", branch):
    #         pos = ""
    #         prev = re.sub(r" |\+-", "", prev)
    #         branch = re.sub(r" |\+-", "", branch)
    #         line = prev + "-" + branch
    #         tag, skip = line.split(":")

    #         if text[ct] in flight_list:
    #             ct += 1
    #             continue

    #         if re.search(r'AUX', tag):
    #             auxiliary.append(True)
    #             # pos = 'AUX'
    #             ct += 1
    #             continue

    #         if len(auxiliary) == 1:
    #             auxiliary = []

    #         if text[ct] in ISK_ADV and re.search(r'ADV', tag):
    #             flag_deg = True

    #             ct += 1
    #             continue

    #         # if re.search(r'ADV', prev):
    #         #     pos = 'ADV'
    #         #     print(pos, text[ct])
    #         #     ct += 1
    #         #     continue

    #         if text[ct] in isk_stop:
    #             ct += 1
    #             continue
            
    #         print(tag, text[ct])
    #         ct += 1
    #         continue
    #     prev = branch

    # for l in text:
    #     if l.lower() in isk_stop:
    #         continue
    #     if l in string.punctuation:
    #         continue
    #     print(l)

    #return [text, polarity_pure, polarity, weight]

def convert_emoji_emoti(input):
    """
    convert_emoji_emoti function replaces all emojis and emoticons in the input with corresponding text descriptions.
    Emoji descriptions are obtained from: https://emojipedia.org
    Emoticon descriptions are obtained from: https://en.wikipedia.org/wiki/List_of_emoticons

    The lexicon for emoji and emoticon are located under '../lexicons/' folder under the name emoji.txt and emoticon.txt correspondingly
    The emoji and emoticon lexicon can be updated by calling the 'update_emoji' or 'update_emoticon' function in the rulebase_test.py file

    : param input: input sentence that will have the emoji converted

    : return: sentence with emoji and emoticons converted into text
    """

    converted_input = input

    tokens = converted_input.split(" ")
    for token in tokens:
        if len(token) > 4 or len(token) == 1:
            continue
        elif token.lower() in STOPWORDS:
            continue
        elif token in emoji_dict:
            converted_input = converted_input.replace(token, (" " + emoji_dict[token] + " "))

    converted_input = converted_input.replace("  ", " ")

    for item in (re.findall(r'[^\w\s]', input)):
        if item in emoji_dict:
            converted_input = converted_input.replace(item, (" " + emoji_dict[item] + " "))
            continue
        converted_input = converted_input.replace(item, "")

    converted_input = converted_input.replace("  ", " ")

    return converted_input

def filter_words(input, sentiment, lang):
    """
    filter_words function calls all other functions relative to processing the input text
    
    : param tokens: list of tokens that requires filtering or certain words
    : param lang: language of the given list of tokens

    : return: cleaned up token, polarity, weight lists
    """

    input = convert_emoji_emoti(input)
      
    if lang == "EN":
        input = spell_correct_eng(input)
        text, polarity_pure, polarity, weight = process_eng(input, sentiment)
    elif lang == "IS":
        text, polarity_pure, polarity, weight = process_isk(input, sentiment)

    return text, polarity_pure, polarity, weight

def open_lexicon(file_name):
    lexicon = {}
    tuning = {}
    path = '../lexicons/'

    with open((path+file_name), encoding='utf-8') as f:
        for line in f:
            word, mean, scores = line.split("\t")
            lexicon[word] = float(mean.strip())
            tuning[word] = [int(x) for x in scores.strip('\n[]').split(', ')]

    f.close()

    return lexicon, tuning

def update_lexicon(df, lang):
    path = '../lexicons/'

    if lang == "EN":
        file_name = 'eng_lexicon.txt'
    elif lang == 'IS':
        file_name = 'isk_lexicon.txt'

    lexicon, tuning = open_lexicon(file_name)

    f = open(path+file_name, 'w', encoding='utf-8')

    for row in df.itertuples(index=False):
        new_word = row.answer_freetext_value
        new_score = row.Sentiment

        if new_score == 1:
            place = 0
        elif new_score == 0:
            place = 1
        else:
            place = 2

        if new_word in tuning:
            tuning[new_word][place] += 1
        else:
            tuning[new_word] = [0, 0, 0]
            tuning[new_word][place] += 1
        
    tuning = sorted(tuning.items())

    for item in tuning:
        mean = ((item[1][0] * 1) + (item[1][2] * -1)) / sum(item[1])
        f.write(item[0] + "\t" + str(mean) + "\t" + str(item[1]) + "\n")

    f.close()

def train(df, lang):
    df = sentiment_to_val(df)
    df = df.apply(lambda x: filter_words(x['answer_freetext_value'], x['Sentiment'], lang), axis=1, result_type='expand')

    df.columns = ['answer_freetext_value', 'Sentiment', 'Deg_Modded', 'Weight']
    del df['Deg_Modded']
    del df['Weight']

    df = df.explode(['answer_freetext_value', 'Sentiment'], ignore_index=True)

    df.dropna(subset = ['answer_freetext_value'], inplace=True)
    df.dropna(subset = ['Sentiment'], inplace=True)

    df.to_excel('isktest.xlsx')

    # update_lexicon(df, lang)

def find_similar(word, lexicon):
    if word in glove_vectors:
        similar_tokens = [x[0] for x in glove_vectors.most_similar(word)]

        for token in similar_tokens:
            if token in lexicon:
                return token

    return word

def find_in_lexicon(tokens, lexicon):
    score = []

    for i in tokens:
        # temp = stem_suffix(i, ENG_SUFFIX, english_words)

        # if temp != None:
        #     i = temp[1]

        if i in lexicon:
            score.append(lexicon[i])
            continue
        
        # similar = find_similar(i, lexicon)
        # if similar in lexicon:
        #     score.append(lexicon[similar])
        #     continue
        
        score.append(0)

    return score

def find_in_tuning(token, tuning):  
    if token in tuning:
        return tuning[token]

    return []

def calculate(input, lang, lexicon):
    input = convert_emoji_emoti(input)

    if lang == "EN":
        input = spell_correct_eng(input, lexicon)
        text, polarity_pure, polarity, weight = process_eng(input, 1)
    elif lang == "IS":
        #text, polarity_pure, polarity, weight = process_eng(input, 1)
        print("Icelandic")

    indiv_score = find_in_lexicon(text, lexicon)
    multiplier = np.multiply(polarity, weight)
    result = np.multiply(indiv_score, multiplier)
    score = sum(result)

    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'

def label(df, lang):
    if lang == "EN":
        file_name = 'eng_lexicon.txt'
    elif lang == "IS":
        file_name = 'isk_lexicon.txt'

    lexicon, skip = open_lexicon(file_name)
    df_sentiment = df.apply(lambda x: calculate(x['answer_freetext_value'], lang, lexicon), axis=1)

    df['Sentiment'] = df_sentiment

    df.to_excel("sentiment_labeled.xlsx")

    return df

def accuracy(df_truth, df_predict):
    """
    accuracy function displays the confusion matrix and calculates precision, recall, f1, f1 micro-average, f1 macro-average scores, and prints it in a tabular format

    : param actual_file: xlsx file with the truth data
    : param test_file: xlsx file with the predicted data

    : return: N/A
    """
    # Open the two files and convert Sentiment column to list
    senti_truth = df_truth['Sentiment'].tolist()
    labels = np.unique(senti_truth)

    senti_predict = df_predict['Sentiment'].tolist()

    # Generate confusion matrix
    #cf_matrix = confusion_matrix(senti_truth, senti_predict, labels=labels)
    #df_matrix = pd.DataFrame(cf_matrix, index=labels, columns=labels)

    #ax = sns.heatmap(df_matrix, annot=True, cmap='Blues', linecolor='white', cbar='True', xticklabels='auto', yticklabels='auto')
    #ax.set(title = "Confusion Matrix",
    #        xlabel = "Predicted Sentiments",
    #        ylabel = "Actual Sentiments")
    #sns.set(font_scale=0.7)

    #plt.show()

    # Calculate precision, recall, accuracy
    precision = precision_score(senti_truth, senti_predict, labels = labels, average = None)
    recall = recall_score(senti_truth, senti_predict, labels = labels, average = None)
    #accuracy = accuracy_score(senti_truth, senti_predict, labels = labels)

    # Calculate f1 score
    f1_gen = f1_score(senti_truth, senti_predict, labels = labels, average = None)
    # Micro average f1 -> calculates positive and negative values globally
    f1_micro = f1_score(senti_truth, senti_predict, labels = labels, average='micro')
    # Macro average f1 -> takes the average of each class's F1 score
    f1_macro = f1_score(senti_truth, senti_predict, labels = labels, average='macro')

    # Compile scores in panda dataframe
    score_compile = np.array([precision, recall, f1_gen])
    f1_average = np.array([f1_micro, f1_macro])
    df_score = pd.DataFrame(data=score_compile, index=['Precision', 'Recall', 'F1'], columns=labels)
    df_average = pd.DataFrame(data=f1_average, index=['F1 Microaverage', 'F1 Macroaverage'], columns=['Scores'])

    # Print dataframe in tabular format
    print(tabulate(df_score, headers='keys', tablefmt='pretty'))
    print(tabulate(df_average, headers='keys', tablefmt='pretty'))

    return f1_micro

def test_lexicon(df, lang):
    df_truth = df.copy()
    df_predict = df.copy()
    del df_predict['Sentiment']

    df_predict = label(df_predict, lang)
    return accuracy(df_truth, df_predict)

def clean_lexicon(lang):
    path = '../lexicons/'

    if lang == "EN":
        file_name = 'eng_lexicon.txt'
    elif lang == 'IS':
        file_name = 'isk_lexicon.txt'

    lexicon, tuning = open_lexicon(file_name)

    keys_to_eliminate = []

    for key, value in tuning.items():
        if len(key) == 1:
            keys_to_eliminate.append(key)
        new_key = stem_prefix(key, ENG_PREFIX, english_words)
        if new_key != None and new_key[1] in tuning:
            keys_to_eliminate.append(key)

            value.reverse()
            new_value = [x + y for x, y in zip(value, tuning[new_key[1]])]

            tuning[new_key[1]] = new_value

    for key in keys_to_eliminate:
        del tuning[key]
    
    keys_to_eliminate = []

    for key, value in tuning.items():
        new_key = stem_suffix(key, ENG_SUFFIX, english_words)
        if new_key != None and new_key[1] in tuning:
            keys_to_eliminate.append(key)

            new_value = [x + y for x, y in zip(value, tuning[new_key[1]])]
            
            tuning[new_key[1]] = new_value

    for key in keys_to_eliminate:
        del tuning[key]

    f = open(path+file_name, 'w', encoding='utf-8')

    for key, value in tuning.items():
        mean = ((value[0] * 1) + (value[2] * -1)) / sum(value)
        f.write(key + "\t" + str(mean) + "\t" + str(value) + "\n")

    f.close()

def tune(input, lang, lexicon):
    input = convert_emoji_emoti(input)

    if lang == "EN":
        input = spell_correct_eng(input, lexicon)
        text, polarity_pure, polarity, weight = process_eng(input, 1)
    elif lang == "IS":
        #text, polarity_pure, polarity, weight = process_isk(input, 1)
        print("Icelandic")

    indiv_score_raw = find_in_lexicon(text, lexicon)
    indiv_score_pol = np.multiply(indiv_score_raw, polarity_pure)

    multiplier = np.multiply(polarity, weight)
    result = np.multiply(indiv_score_raw, multiplier)
    score = sum(result)

    predicted = 0

    if score > 0:
        predicted = 1
    elif score < 0:
        predicted = -1

    #return predicted, score, text, indiv_score_raw, indiv_score_pol, weight
    return predicted, text, polarity_pure, indiv_score_pol

def tune_lexicon(df, lang):
    tune_copy = df.copy(deep=True)

    if lang == "EN":
        file_name = 'eng_lexicon.txt'
    elif lang == "IS":
        file_name = 'isk_lexicon.txt'

    df = sentiment_to_val(df)

    lexicon, tuning = open_lexicon(file_name)
    df_tune = df.apply(lambda x: tune(x['answer_freetext_value'], lang, lexicon), axis=1, result_type='expand')
    df_tune.columns = ['predicted', 'answer_freetext_value', 'polarity', 'score_pol']

    del df['answer_freetext_value']
    df = pd.concat([df, df_tune], axis=1)

    df = df.drop(df[df['Sentiment'] == df['predicted']].index)
    del df['predicted']

    df = df.explode(['answer_freetext_value', 'polarity', 'score_pol'])
    df = df.reset_index(drop=True)

    df['sign'] = df['score_pol'].apply(lambda x: np.sign(x))
    df = df.drop(df[df['Sentiment'] == df['sign']].index)
    df.dropna(subset = ['answer_freetext_value'], inplace=True)

    df['Sentiment'] = df['Sentiment'] * df['polarity']
    del df['sign']
    del df['score_pol']
    del df['polarity']

    update_lexicon(df, lang)
