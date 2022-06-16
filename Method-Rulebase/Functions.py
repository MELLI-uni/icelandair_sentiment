from tokenize import String
from itertools import groupby
from itertools import islice
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

import regex as re
import xlwings as xws

import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS

from tabulate import tabulate

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import spacy
from reynir import Greynir
from reynir_correct import check_single

eng_pattern = r'\s[Bb]ut\.*,*\s|\s[Hh]owever\.*,*\s|\s[Ee]xcept\.*,*\s'
isk_pattern = r'\s[Ee][n]\.*,*\s|\s[Þþ]ó\.*,*\s|\s[Nn]ema\.*,*\s|\s[Hh]ins vegar\.*,*\s|\sfyrir utan\.*,*\s'

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

deg_adv = []

# regex pattern for flight names
regex_plane = r'(A3\d{2}(-\d{3})?)|(7\d7(-\d{3})?)|FI\d+'

emoji_dict = {}
with open('../lexicons/emoji.txt', encoding='utf-8') as f:
    for line in islice(f, 1, None):
        [key, value] = line.split("\t")
        emoji_dict[key] = value
with open('../lexicons/emoticon.txt', encoding='utf-8') as f:
    for line in f:
        [key, value] = line.split("\t")
        emoji_dict[key] = value

isk_stop = []
with open('../lexicons/isk_stop.txt', encoding='utf-8') as f:
    for line in f:
        isk_stop.append(line.strip())

FLIGHT = []
with open('../lexicons/destination.txt', encoding='utf-8') as f:
    for line in f:
        FLIGHT.append(line)

nlp = spacy.load("en_core_web_sm")
g = Greynir()

def init(file_name, sheet_name):
    """
    init function launches a password-protected excel file for the user to open and changes it into a datafram

    : param file_name: location/name of the excel file to open
    : param sheet_name: name of the sheet to open

    : return: data in pandas dataframe
    """

    wb = xws.Book(file_name)
    sheet = wb.sheets[sheet_name].used_range

    df = sheet.options(pd.DataFrame, index=False, header=True).value
    return(df)

def clean(df, lang):
    """
    clean function eliminates columns that are not needed, rows with no freetext or sentiment, lowercase and strips
    trailing blank spaces in the sentiment column, and separates the sentiment into three boolean columns (pos, neg, neu)

    : param df: panda dataframe

    : return: panda dataframe with [id(int), freetext(String), Positive(bool), Negative(bool), Neutral(bool)]

    input: panda_df
    output: cleaned panda_df with (id, freetext, Sentiment)
    """

    header = list(df.columns)

    # Leave only id, freetext, Sentiment column
    to_leave = ['id', 'answer_freetext_value', 'Sentiment']

    for h in header:
        if h not in to_leave:
            del df[h]

    # Lowercase & Strip trailing blanks for sentiment
    # Change sentiment column into list by spliting at non-word component
    df['Sentiment'] = df['Sentiment'].str.lower().str.strip()
    df['Sentiment'] = list(df['Sentiment'].str.split(r'\W+'))

    # Drop all rows with either empty freetext or emtpy sentiment
    df.dropna(subset = ['answer_freetext_value'], inplace=True)
    df.dropna(subset = ['Sentiment'], inplace=True)

    # Find all rows with multiple sentiments and store it to separate dataframe
    df_temp = df.loc[(df['Sentiment']).str.len() > 1]
    df.drop(df_temp.index, inplace=True)
    df = df.explode(['Sentiment'])

    #df_temp.to_excel("multi.xlsx")      # LINE TO DELETE

    df_count = df_temp[['Sentiment', 'answer_freetext_value']].copy()   # Create a copy of the dataframe with multiple sentiments
    df_count['answer_freetext_value'] = df_count['answer_freetext_value'].str.replace(r'\s*([Tt]hank [Yy]ou)\.*\s*','', regex=True)  # Replace all 'Thank you.' to blank space
    df_count['answer_freetext_value'] = df_count['answer_freetext_value'].str.replace(r'\s*[Yy]es\.*\,*\s*','', regex=True)  # Replace all 'Yes.' to blank space

    df_count['answer_freetext_value'] = df_count['answer_freetext_value'].str.replace(r'\s*[Tt]akk\.\s*','', regex=True)
    df_count['answer_freetext_value'] = df_count['answer_freetext_value'].str.replace(r'\s*[Jj]á\.*\,*\s*','', regex=True)
    
    df_count['Sentiment'] = df_count['Sentiment'] # Count number of sentiments
    df_count['New Line'] = df_count['answer_freetext_value'].str.split(r'\n+')    # Count number of sentences separated by new line

    if lang == "IS":
        negating_conjugation = isk_pattern
    else:
        negating_conjugation = eng_pattern
    
    df_count['Negating'] = df_count['answer_freetext_value'].str.split(negating_conjugation)    # Count number of sentences separated by negating conjugations
    # TODO: Investigate with \p{L} -> why is it not workiiinnnggggggg
    df_count['Period'] = df_count['answer_freetext_value'].str.strip(r'\W+').str.split(r'(?<=[a-zA-ZÁáÐðÉéÍíÓóÚúÝýÞþÆæÖö])\.(?=\s*[a-zA-ZÁáÐðÉéÍíÓóÚúÝýÞþÆæÖö])') # Count number of sentences separated by period

    df_count['Punct'] = df_count['answer_freetext_value'].str.split(r',|;') # Count number of sentences separated by comma or semicolon
    df_count['Elim'] = df_temp['Sentiment'].apply(lambda x: [i[0] for i in groupby(x)])   # Remove consecutive duplicates in sentiment list

    #TODO: Determine priority or if accuracy of program is high enough, train this portion with transformer?!?!
    df_count.loc[(df_count['Sentiment'].str.len() == df_count['New Line'].str.len()), 'Type'] = 'New Line'
    df_count.loc[(df_count['Sentiment'].str.len() == df_count['Negating'].str.len()) & (df_count['Type'].isnull()), 'Type'] = 'Negating'
    df_count.loc[(df_count['Sentiment'].str.len() == df_count['Period'].str.len()) & (df_count['Type'].isnull()), 'Type'] = 'Period'

    df_count.loc[(df_count['Elim'].str.len() == df_count['New Line'].str.len()), 'Type'] = 'New Line'
    df_count.loc[(df_count['Elim'].str.len() == df_count['New Line'].str.len()), 'Sentiment'] = df_count['Elim']    # Replace sentiment label with duplicate removed list

    df_count.loc[(df_count['Elim'].str.len() == df_count['Negating'].str.len()) & (df_count['Type'].isnull()), 'Sentiment'] = df_count['Elim']    # Replace sentiment label with duplicate removed list
    df_count.loc[(df_count['Elim'].str.len() == df_count['Negating'].str.len()) & (df_count['Type'].isnull()), 'Type'] = 'Negating'
    del df_count['Elim']

    df_count.loc[(df_count['Period'].str.len() == 1) & (df_count['Sentiment'].str.len() == df_count['Punct'].str.len()) & (df_count['Type'].isnull()), 'Type'] = 'Punct'

    df_temp['Sentiment'] = df_count['Sentiment']
    df_temp.loc[(df_count['Type'] == 'New Line'), 'answer_freetext_value'] = df_count['New Line'][df_count['Type'] == 'New Line']
    df_temp.loc[(df_count['Type'] == 'Negating'), 'answer_freetext_value'] = df_count['Negating'][df_count['Type'] == 'Negating']
    df_temp.loc[(df_count['Type'] == 'Period'), 'answer_freetext_value'] = df_count['Period'][df_count['Type'] == 'Period']
    df_temp.loc[(df_count['Type'] == 'Punct'), 'answer_freetext_value'] = df_count['Punct'][df_count['Type'] == 'Punct']
    df_temp['Type'] = df_count['Type']
    #df_temp.to_excel("multi_det.xlsx")      # LINE TO DELETE

    df_temp.dropna(subset = ['Type'], inplace=True)
    del df_temp['Type']

    df_temp = df_temp.explode(['Sentiment', 'answer_freetext_value'])
    #df_temp.to_excel("multi_explode.xlsx")       # LINE TO DELETE

    df = pd.concat([df, df_temp], ignore_index=True, sort=False)

    df.loc[df['Sentiment'] == 'positive', 'Sentiment'] = 0.25
    df.loc[df['Sentiment'] == 'negative', 'Sentiment'] = -0.25
    df.loc[df['Sentiment'] == 'neutral', 'Sentiment'] = 0
    #del df['Sentiment']

    return df

def eng_process(input, sentiment):
    text = []
    tag = []
    score = []
    
    token_score = sentiment

    doc = nlp(input)
    
    for token in doc:
        if (
            (token.lemma_ in ENG_ADV and token.tag_ == 'RB')
            or token.text in FLIGHT
            or token.text in NEUTRAL_SKIP
            or token.lemma_ == 'be'
            or re.findall(r'\,|NNP|NNPS|RP|SYM|TO|UH|WDT|WP|WP$|WRB|LS|IN|FW|EX|DT|CC|CD|MD', token.tag_)   # test with NN and NNS & test with MD
        ):
            continue
        
        if re.findall(r'\.', token.tag_):
            token_score = sentiment
            continue

        if re.findall(r'\W', token.text):
            continue

        if token.lemma_ in ENG_NEG:
            if tag and (tag[-1] == 'MD' or text[-1] == 'do' or text[-1] == 'have'):
                del text[-1]
                del tag[-1]
                del score[-1]
            token_score = -token_score
            continue

        text.append(token.lemma_)
        tag.append(token.tag_)

        if token.tag_ == 'JJ':
            score.append(token_score * 4)
        else:
            score.append(token_score)

    return [text, score]

def eng_process2(input, sentiment):
    text = []
    tag = []
    score = []
    
    token_score = sentiment
    amplify = 0

    doc = nlp(input)
    
    for token in doc:
        if (
            token.text in FLIGHT
            or token.text in NEUTRAL_SKIP
            or token.lemma_ == 'be'
            or re.findall(r'\,|NNP|NNPS|RP|SYM|TO|UH|WDT|WP|WP$|WRB|LS|IN|FW|EX|DT|CC|CD|MD', token.tag_)   # test with NN and NNS & test with MD
        ):
            continue
        
        if re.findall(r'\.', token.tag_):
            token_score = sentiment
            continue

        if re.findall(r'\W', token.text):
            continue

        if token.lemma_ in ENG_ADV:
            amplify = ENG_ADV[token.lemma_]
            continue

        if token.lemma_ in ENG_NEG:
            if tag and (tag[-1] == 'MD' or text[-1] == 'do' or text[-1] == 'have'):
                del text[-1]
                del tag[-1]
                del score[-1]
            token_score = -token_score
            continue

        text.append(token.lemma_)
        tag.append(token.tag_)

        if token_score != sentiment:
            amplify = -amplify

        if token.tag_ == 'JJ':
            score.append(amplify + token_score * 4)
        else:
            score.append(amplify + token_score)
        amplify = 0

    return [text, score]

def cleaning(input, senti):
    doc = nlp(input)
    auxiliary = []
    degree = ""
    negate = False
    label = senti
    comp_neg = False

    tokens = []
    score = []

    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.dep_, label)
        if token.pos_ == 'PUNCT' or token.pos_ == 'SPACE':
            continue

        elif token.dep_ == 'cc':
            comp_neg = False

        elif token.pos_ == 'ADJ':
            tokens.append(token.lemma_)
            if negate == True:
                negate == False
                score.append(-label)
                comp_neg = True
                continue
            if len(degree) != 0:
                degree_adv.append(degree)
                degree = ""
            score.append(label)
            continue

        elif token.pos_ == 'VERB':
            if negate == True:
                tokens.append(token.lemma_)
                score.append(-label)
                negate = False
                comp_neg = True
                continue
            elif token.dep_ == 'ROOT':
                tokens.append(token.lemma_)
                score.append(label)
                continue

        auxiliary = []

    return [tokens, score]

def isk_process(input, sentiment):
    text = []
    score = []

    lines = input.strip(".!?").split(".")

    for sentence in lines:
        sent = check_single(sentence)
        if sent is None:
            continue

        sent = sent.tidy_text
        doc = g.parse_single(sent)

        token_count = 0
        token_score = sentiment
        token_lemmas = doc.lemmas
        token_tags = doc.categories

        if token_lemmas:
            for token in token_lemmas:
                if (
                    token in ISK_ADV
                    or token in isk_stop
                    or token == 'vera'
                    or token in FLIGHT
                    or re.findall(r'\p{S}|\p{Ps}|\p{Pe}|\p{Pi}|\p{Pf}|\p{Pc}|\p{Po}', token)
                    or re.findall(r'.*fn|person|entity|gata|to|töl|st.*|uh|nhm|gr|fs|fyrirtæki', token_tags[token_count])
                ):
                    token_count += 1
                    continue

                if text and token in ISK_NEG:
                    if (token_count != len(token_tags)) and (token_tags[token_count + 1] == 'so' or token_tags[token_count + 1] == 'lo'):
                        del text[-1]
                        del score[-1]

                    else:
                        score[-1] = -score[-1]

                    token_score = -token_score
                    token_count += 1
                    continue

                if token_tags[token_count] == 'lo':
                    score.append(token_score * 2)
                else:
                    score.append(token_score)
                token_count += 1

                text.append(token)

    return [text, score]

def process(df, lang):
    if lang == "EN":
        df = df.apply(lambda x: eng_process(x['answer_freetext_value'], x['Sentiment']), axis=1, result_type='expand')
    elif lang == "IS":
        df = df.apply(lambda x: isk_process(x['answer_freetext_value'], x['Sentiment']), axis=1, result_type='expand')

    df.columns = ['answer_freetext_value', 'Sentiment']
    df = df.explode(['answer_freetext_value', 'Sentiment'], ignore_index=True)

    df.dropna(subset = ['answer_freetext_value'], inplace=True)
    df.dropna(subset = ['Sentiment'], inplace=True)

    return df

# Intensify adjectives and adverbs when calculating scores

# If not is placed in front of an adjective then change the storing sentiment when developing lexicon list

def make_dict(lexicon):
    lex_dict = {}

    loc = './lexicons/' + lexicon

    with open(loc, encoding='utf-8') as f:
        for line in f:
            [key, value, skip] = line.split("\t")
            lex_dict[key] = float(value)

    return lex_dict

def find_in_dict(tokens, dict):
    score = []

    for i in tokens:
        if i in dict:
            score.append(dict[i])
        else:
            score.append(0)

    return score

def calculate(text, lang, lexi_dict):
    if lang == "EN":
        [text, multiplier] = eng_process(text, 1)
    elif lang == "IS":
        [text, multiplier] = isk_process(text, 1)
    
    indiv_scores = find_in_dict(text, lexi_dict)
    product = np.multiply(multiplier, indiv_scores)
    score = sum(product)

    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'

def label(df):
    lexi_dict = make_dict('eng_lexicon.txt')
    df_senti = df.apply(lambda x: calculate(x['answer_freetext_value'], "EN", lexi_dict), axis=1)
    df['Sentiment'] = df_senti

    df.to_excel("guess.xlsx")


################################################################

#TODO: MAKE IT SO THAT IT ONLY DIVIDES AT NEW LINE, BY EVERY PERIOD, AND AT NEGATING CONJUGATIONS
def leave_text(df, lang):
    """
    clean function eliminates columns that are not needed, rows with no freetext or sentiment, lowercase and strips
    trailing blank spaces in the sentiment column, and separates the sentiment into three boolean columns (pos, neg, neu)

    : param df: panda dataframe

    : return: panda dataframe with [id(int), freetext(String), Positive(bool), Negative(bool), Neutral(bool)]

    input: panda_df
    output: cleaned panda_df with (id, freetext, Sentiment)
    """

    header = list(df.columns)

    # Leave only id, freetext, Sentiment column
    to_leave = ['id', 'answer_freetext_value', 'Sentiment']

    for h in header:
        if h not in to_leave:
            del df[h]

    # Lowercase & Strip trailing blanks for sentiment
    # Change sentiment column into list by spliting at non-word component
    df['Sentiment'] = df['Sentiment'].str.lower().str.strip()
    df['Sentiment'] = list(df['Sentiment'].str.split(r'\W+'))

    # Drop all rows with either empty freetext or emtpy sentiment
    df.dropna(subset = ['answer_freetext_value'], inplace=True)
    df.dropna(subset = ['Sentiment'], inplace=True)

    # Find all rows with multiple sentiments and store it to separate dataframe
    df_temp = df.loc[(df['Sentiment']).str.len() > 1]
    df.drop(df_temp.index, inplace=True)
    df = df.explode(['Sentiment'])

    #df_temp.to_excel("multi.xlsx")      # LINE TO DELETE

    df_count = df_temp[['Sentiment', 'answer_freetext_value']].copy()   # Create a copy of the dataframe with multiple sentiments
    df_count['answer_freetext_value'] = df_count['answer_freetext_value'].str.replace(r'\s*([Tt]hank [Yy]ou)\.*\s*','', regex=True)  # Replace all 'Thank you.' to blank space
    df_count['answer_freetext_value'] = df_count['answer_freetext_value'].str.replace(r'\s*[Yy]es\.*\,*\s*','', regex=True)  # Replace all 'Yes.' to blank space

    df_count['answer_freetext_value'] = df_count['answer_freetext_value'].str.replace(r'\s*[Tt]akk\.\s*','', regex=True)
    df_count['answer_freetext_value'] = df_count['answer_freetext_value'].str.replace(r'\s*[Jj]á\.*\,*\s*','', regex=True)
    
    df_count['Sentiment'] = df_count['Sentiment'] # Count number of sentiments
    df_count['New Line'] = df_count['answer_freetext_value'].str.split(r'\n+')    # Count number of sentences separated by new line

    if lang == "IS":
        negating_conjugation = isk_pattern
    elif lang == "EN":
        negating_conjugation = eng_pattern
    
    df_count['Negating'] = df_count['answer_freetext_value'].str.split(negating_conjugation)    # Count number of sentences separated by negating conjugations
    # TODO: Investigate with \p{L} -> why is it not workiiinnnggggggg
    df_count['Period'] = df_count['answer_freetext_value'].str.strip(r'\W+').str.split(r'(?<=[a-zA-ZÁáÐðÉéÍíÓóÚúÝýÞþÆæÖö])\.(?=\s*[a-zA-ZÁáÐðÉéÍíÓóÚúÝýÞþÆæÖö])') # Count number of sentences separated by period
    #df_count['Word'] = df_count['answer_freetext_value'].str.split()  # Count number of words in the sentence
    #df_count['Word'] = df_count['Word'].apply(lambda x: [' '.join(i.tolist()) for i in (np.array_split(np.array(x), 2))])
    #df_count['Wordc'] = df_count.apply(lambda x: np.array(x['Word']))

    df_count['Punct'] = df_count['answer_freetext_value'].str.split(r',|;') # Count number of sentences separated by comma or semicolon
    df_count['Elim'] = df_temp['Sentiment'].apply(lambda x: [i[0] for i in groupby(x)])   # Remove consecutive duplicates in sentiment list

    #TODO: Determine priority or if accuracy of program is high enough, train this portion with transformer?!?!
    df_count.loc[(df_count['Sentiment'].str.len() == df_count['New Line'].str.len()), 'Type'] = 'New Line'
    df_count.loc[(df_count['Sentiment'].str.len() == df_count['Negating'].str.len()) & (df_count['Type'].isnull()), 'Type'] = 'Negating'
    df_count.loc[(df_count['Sentiment'].str.len() == df_count['Period'].str.len()) & (df_count['Type'].isnull()), 'Type'] = 'Period'

    df_count.loc[(df_count['Elim'].str.len() == df_count['New Line'].str.len()), 'Type'] = 'New Line'
    df_count.loc[(df_count['Elim'].str.len() == df_count['New Line'].str.len()), 'Sentiment'] = df_count['Elim']    # Replace sentiment label with duplicate removed list

    df_count.loc[(df_count['Elim'].str.len() == df_count['Negating'].str.len()) & (df_count['Type'].isnull()), 'Sentiment'] = df_count['Elim']    # Replace sentiment label with duplicate removed list
    df_count.loc[(df_count['Elim'].str.len() == df_count['Negating'].str.len()) & (df_count['Type'].isnull()), 'Type'] = 'Negating'
    del df_count['Elim']

    df_count.loc[(df_count['Period'].str.len() == 1) & (df_count['Sentiment'].str.len() == df_count['Punct'].str.len()) & (df_count['Type'].isnull()), 'Type'] = 'Punct'

    df_temp['Sentiment'] = df_count['Sentiment']
    df_temp.loc[(df_count['Type'] == 'New Line'), 'answer_freetext_value'] = df_count['New Line'][df_count['Type'] == 'New Line']
    df_temp.loc[(df_count['Type'] == 'Negating'), 'answer_freetext_value'] = df_count['Negating'][df_count['Type'] == 'Negating']
    df_temp.loc[(df_count['Type'] == 'Period'), 'answer_freetext_value'] = df_count['Period'][df_count['Type'] == 'Period']
    df_temp.loc[(df_count['Type'] == 'Punct'), 'answer_freetext_value'] = df_count['Punct'][df_count['Type'] == 'Punct']
    df_temp['Type'] = df_count['Type']
    #df_temp.to_excel("multi_det.xlsx")      # LINE TO DELETE

    df_temp.dropna(subset = ['Type'], inplace=True)
    del df_temp['Type']

    df_temp = df_temp.explode(['Sentiment', 'answer_freetext_value'])
    #df_temp.to_excel("multi_explode.xlsx")       # LINE TO DELETE

    df = pd.concat([df, df_temp], ignore_index=True, sort=False)

    df.to_excel("truth.xlsx")

    del df['Sentiment']
    df.to_excel("test.xlsx")

    return df

def accuracy(actual_file, test_file):
    """
    accuracy function displays the confusion matrix and calculates precision, recall, f1, f1 micro-average, f1 macro-average scores, and prints it in a tabular format

    : param actual_file: xlsx file with the truth data
    : param test_file: xlsx file with the predicted data

    : return: N/A
    """
    # Open the two files and convert Sentiment column to list
    df_truth = pd.read_excel(actual_file)
    senti_truth = df_truth['Sentiment'].tolist()
    labels = np.unique(senti_truth)

    df_predict = pd.read_excel(test_file)
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
