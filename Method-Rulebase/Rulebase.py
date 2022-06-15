import regex as re
import xlwings as xws
import string
import emoji

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
from nltk.corpus import words
from nltk.corpus import wordnet as wn
from nltk.util import ngrams
from nltk.metrics.distance import jaccard_distance
from nltk.metrics.distance import edit_distance

from reynir import Greynir
from reynir_correct import check_single

eng_spacy = spacy.load('en_core_web_sm')
isk_greynir = Greynir()

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

english_words = list(wn.words()) + words.words()

eng_negating = r'\s[Bb]ut\.*,*\s|\s[Hh]owever\.*,*\s'
isk_negating = r'\s[Ee]n\.*,*\s|\s[Nn]ema\.*,*\s'

ADV_VAL = 1
A_INC = ADV_VAL
A_DEC = -ADV_VAL

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

ISK_NEG = ["ekki"]
#ISK_NEG = ["aldrei", "ekkert", "ekki", "enginn", "hvergi", "hvorki", "ne", "neibb", "neitt"]
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

def stem_prefix(word, prefixes, roots):
    original_word = word

    for prefix in sorted(prefixes, key=len, reverse=True):
        word, nsub = re.subn(prefix, "", word)
        if nsub > 0 and word in roots:
            return prefixes[prefix] + word
    
    return original_word

def stem_suffix(word, suffixes, roots):
    original_word = word

    for suffix in sorted(suffixes, key=len, reverse=True):
        word, nsub = re.subn(suffix, "", word)
        if nsub > 0 and word in roots:
            return word
    
    return original_word

def spell_correct_eng(word):
    """
    Reference: https://socialnetwork.readthedocs.io/en/latest/spell-check.html
    """
    temp = [(jaccard_distance(
            set(ngrams(word, 2)),
            set(ngrams(w, 2))), w)
            for w in english_words if w[0]==word[0]]
    
    return sorted(temp, key = lambda val: val[0])[0][1]

def lemmatize_eng(input):
    return " ".join([token.lemma_ for token in eng_spacy(input) if token.text not in string.punctuation])

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

def total_weight(num_levels, num_degree):
    sum = (num_levels * (num_levels + 1)) / 2
    sum += num_degree

    return sum

def weight_tokens():
    return

def process_eng(input, sentiment):
    doc = eng_spacy(input)

    text = []
    tag = []
    polarity = []

    token_score = sentiment
    auxiliary = []
    negate = False
    comp_neg = False

    score_deg = 0
    flag_deg = False

    for token in doc:
        # if token.text not in english_words:
        #     continue

        if token.pos_ == 'AUX':
            auxiliary.append(True)
            continue

        if len(auxiliary) == 1:
            auxiliary = []

        if len(auxiliary) >= 2:
            if(token.pos_ == 'ADJ'):
                auxiliary = []
                text.append(token.lemma_)
                polarity.append(-token_score)
                tag.append(token.pos_)
                comp_neg = True
            continue

        if token.dep_ == 'cc':
            comp_neg = False
            continue

        if token.dep_ == 'prep':
            continue

        if token.dep_ == 'neg':
            negate = True
            continue

        if negate == True:
            text.append(token.lemma_)
            polarity.append(-token_score)
            tag.append(token.pos_)
            negate = False
            continue

        if token.dep_ == 'mark':
            if comp_neg == True:
                token_score = -token_score
            continue

        # TODO: Find a way to indicate that this is for the following adj
        if token.lemma_ in ENG_ADV and token.pos_ == 'ADV':
            score_deg += ENG_ADV[token.lemma_]
            flag_deg = True
            continue

        if token.pos_ == r'[\.\?!]':
            token_score = sentiment
            continue

        if token.pos_ == 'NOUN' and token.lemma_ in text:
            continue

        if re.findall(r'ADP|CONJ|CCONJ|DET|INTJ|NUM|PART|PRON|PROPN|PUNCT|SCONJ|SYM|X|SPACE', token.pos_):
            continue

        if token.lemma_ in STOPWORDS:
            continue

        text.append(token.lemma_)
        polarity.append(token_score)
        tag.append(token.pos_)
        #print(token.text, token.lemma_, token.tag_, token.pos_, token.dep_, token_score)

    list_tag = Counter(tag).keys()
    weights = (np.abs(sentiment) * len(text)) / total_weight(len(list_tag), score_deg)

    print(text)
    print(polarity)
    print(tag)
    print(weights)

    return input

def sample_isk():
    my_text = "Ég hataði flugið vegna þess að flugvélin var svo heit"

    job = isk_greynir.submit(my_text)

    tree_temp = []
    tree = []
    lemmas = []

    # Iterate through sentences and parse each one
    for sent in job:
        sent.parse()

        if sent.tree is None:
            print(False)
            continue
            # for t in sent.tokens:
            #     token_list.append(t.txt)

        print(sent.tree.view)
        tree_temp = sent.tree.view.split("\n")

    for levels in tree_temp[2:]:
        elim_indic = re.sub('+-', '', levels)
        tree.append(re.subn('  ', '', elim_indic))

    print(tree)

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

def filter_words(tokens, lang):
    """
    filter_words function replaces all airport related information from the given list of tokens
    It also replaces all information that seems like plane names or reference numbers
    Then the function calls the processing function based on the language, which would provide additional cleaning onto the tokens
    
    : param tokens: list of tokens that requires filtering or certain words
    : param lang: language of the given list of tokens

    : return: cleaned up token
    """
    
    return tokens

def open_lexicon(file_name):
    lexicon = {}
    path = '../lexicons/'

    with open((path+file_name), encoding='utf-8') as f:
        for line in f:
            word, score = line.split("\t")
            lexicon[word] = float(score.strip())

    # format word score pos neg neu

    return lexicon

def update_lexicon(df):
    return

def find_in_lexicon(tokens, lexicon):
    score = []

    for i in tokens:
        if i in lexicon:
            score.append(lexicon[i])
            continue
        score.append(0)

    return score

def calculate(input, lang, lexicon):
    if lang == "EN":
        tokens = []
    elif lang == "IS":
        tokens = []

    indiv_score = find_in_lexicon(tokens, lexicon)

    score = 0

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

    lexicon = open_lexicon(file_name)
    df_sentiment = df.apply(lambda x: calculate(x['answer_freetext_value'], lang, lexicon), axis=1)

    df['Sentiment'] = df_sentiment

    df.to_excel("sentiment_labeled.txt")

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

    ##### TO BE DELETED
    df = df_truth.copy()
    df['Guess'] = df_predict['Sentiment']

    df['Inc'] = np.where(df['Sentiment'] != df['Guess'], 'INCORRECT', '')
    df.to_excel("Checking.xlsx")
    #####

    # Generate confusion matrix
    cf_matrix = confusion_matrix(senti_truth, senti_predict, labels=labels)
    df_matrix = pd.DataFrame(cf_matrix, index=labels, columns=labels)

    ax = sns.heatmap(df_matrix, annot=True, cmap='Blues', linecolor='white', cbar='True', xticklabels='auto', yticklabels='auto')
    ax.set(title = "Confusion Matrix",
            xlabel = "Predicted Sentiments",
            ylabel = "Actual Sentiments")
    sns.set(font_scale=0.7)

    plt.show()

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

def test_lexicon(df, lang):
    df_truth = df.copy()
    df_predict = df.copy()
    del df_predict['Sentiment']

    df_predict = label(df_predict, lang)
    accuracy(df_truth, df_predict)

def tuning(df_truth, lang):
    if lang == "EN":
        file_name = 'eng_lexicon.txt'
    elif lang == "IS":
        file_name = 'isk_lexicon.txt'

    lexicon = open_lexicon(file_name)
    


    return 

def tune_lexicon(df, lang):
    df_truth = df.copy()

    tuning(df_truth, lang)