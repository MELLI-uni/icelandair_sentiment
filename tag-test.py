import spacy
import regex as re
from reynir import Greynir
from reynir_correct import check_single
from itertools import islice

ADV_VAL = 0.25
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

# ENG_NEG = ["aint", "ain't", "arent", "aren't", "cannot", "cant", "can't", "darent", "daren't", "didnt", "didn't", "doesnt", "doesn't", "don't", "dont", "hadnt", "hadn't", "hasnt", 
#             "hasn't", "have-not", "havent", "haven't", "isnt", "isn't", "mightnt", "mightn't", "neednt", "needn't", "neither", "never", "none", "nope", "nor", "not", "nothing", 
#             "nowhere", "shant", "shan't", "shouldnt", "shouldn't", "wasnt", "wasn't", "wont", "won't", "wouldnt", "wouldn't", 
#             ]

ENG_NEG = ["neither", "never", "none", "nope", "nor", "not", "nothing", "nowhere"]
ISK_NEG = ["aldrei", "ekkert", "ekki", "enginn", "hvergi", "hvorki", "ne", "neibb", "neitt"]

emoji_dict = {}
with open('./lexicons/emoji.txt', encoding='utf-8') as f:
    for line in islice(f, 1, None):
        [key, value] = line.split("\t")
        emoji_dict[key] = value
with open('./lexicons/emoticon.txt', encoding='utf-8') as f:
    for line in f:
        [key, value] = line.split("\t")
        emoji_dict[key] = value

isk_stop = []
with open('./lexicons/isk_stop.txt', encoding='utf-8') as f:
    for line in f:
        isk_stop.append(line.strip())

nlp = spacy.load("en_core_web_sm")
g = Greynir()

text = "Very poor experience given the price paid. Many elements need sorting out. Probably too many, so won't be travelling with Iceland Air again: I haven't like the flight"

# if word in degree adverb list (and is tag is adv) remove from storing
# if word has preceding negation, negate the storing value
# if word is an adjective save as value *2
# if word in flight destination dictionary eliminate
# if word is NNP eliminate
# if word is period eliminate (do not use tag)
# if word is exclamation point count number of exclamation points (in case of deving lexicon list eliminate)
# if word is DT eliminate
# if word in emoji.txt or emoticon.txt dictionary change to words (LAST STEP)

def processed_list(input, sentiment):
    text = []
    tag = []
    score = []
    
    token_score = sentiment

    doc = nlp(input)
    
    for token in doc:
        if (
            (token.lemma_ in ENG_ADV and token.tag_ == 'RB')
            or token.lemma_ == 'be'
            or re.findall(r'\,|NNP|NNPS|DT|IN|PRP|PRP$', token.tag_)
        ):
            continue
        
        if re.findall(r'\.', token.tag_):
            token_score = sentiment
            continue

        if re.findall(r'\W', token.text):
            continue

        if token.lemma_ in ENG_NEG:
            if tag[-1] == 'MD' or text[-1] == 'do' or text[-1] == 'have':
                del text[-1]
                del tag[-1]
                del score[-1]
            token_score = -token_score
            continue

        text.append(token.lemma_)
        tag.append(token.tag_)

        if token.tag_ == 'JJ':
            score.append(token_score * 2)
        else:
            score.append(token_score)

    return text, score

# labeling of token.tag_: https://cs.nyu.edu/~grishman/jet/guide/PennPOS.html
# labeling of token.pos_: https://universaldependencies.org/u/pos/

# regex: https://www.regular-expressions.info/unicode.html#prop
def isk_process(input, sentiment):
    text = []
    score = []

    lines = input.strip(".!?").split(".")

    for sentence in lines:
        sent = check_single(sentence).tidy_text
        doc = g.parse_single(sent)

        token_count = 0
        token_score = sentiment
        token_lemmas = doc.lemmas
        token_tags = doc.categories

        print(token_lemmas)
        print(token_tags)

        for token in token_lemmas:
            if (
                token in ISK_ADV
                or token in isk_stop
                or token == 'vera'
                or re.findall(r'\p{S}|\p{Ps}|\p{Pe}|\p{Pi}|\p{Pf}|\p{Pc}|\p{Po}', token)
                or re.findall(r'.*fn|person|entity|gata|to|töl|st.*|uh|nhm|gr|fs|fyrirtæki', token_tags[token_count])
            ):
                token_count += 1
                continue

            if token in ISK_NEG:
                if token_tags[token_count + 1] == 'so' or token_tags[token_count + 1] == 'lo':
                    del text[-1]
                    del score[-1]

                else:
                    score[-1] = -score[-1]

                token_score = -token_score
                token_count += 1
                continue

            text.append(token)

            if token_tags[token_count] == 'lo':
                score.append(token_score * 2)
            else:
                score.append(token_score)
            token_count += 1

    return [text, score]

    # doc = g.parse(input)
    # for sentence in doc["sentences"]:
    #     # sent = check_single(sentence)
    #     # print(sent.tidy_text)

    #     print(type(sentence))

        # print(sent.lemmas)
        # print(sent.categories)
    # print(s.lemmas)
    # print(s.categories)

# is_text = "Hálftíma seinkun á brottför. Sátum út i vel en vorum ekki látin vita hvers vegna seinkun átti sér stað. Hefði kunnað að meta að fá að vita af seinkun og þá hvers vegna"
# text_list = is_text.split(".")

# isk_text = ["Hann myndi ekki klára", "Hann er ekki búinn", "Hann kláraði ekki", "hann kláraði ekki sinn hlut", "hann kláraði ekki að byggja húsið", "hann gaf okkur ekki mat"]
# for l in isk_text:
#     s = g.parse_single(l)
#     print(s.lemmas)
#     print(s.categories)

text = "Greinilegt var að þarna var verið að setja tvö flug saman. Þegar við komum um borð var fullt í hólfum fyrir ofan sæti á Saga Class, við fjölskyldan gátum ekki setið saman þar sem búið var að láta eitt af okkar sætum til hjóna með lítið barn, flugfreyjan var hvorki gott viðmót né þjónustulunduð. Verð að segja að þegar við kaupum rándýra miða á Saga Class (sem við gerum ansi oft), þá býst maður við meiru!!"

job = g.submit(text)

for sent in job:
    sent.parse()
    for f in sent.tree.view:
        print(sent.tree.view)

# isk_process(text, -0.5)

# print(isk_process(text, -0.5))


# idx = s.lemmas.index('ekki')
# print(s.categories[idx-1] + " " + s.categories[idx])

#icelandic negation: (something) ekki

# Greynir_category = {'no':'NN', 'so':'VB', 'lo':'JJ', 'fs':'IN', 'ao':'RB', 'eo':'RB'
                    
# }

"""
Greynir categories
no = noun
so = verb
lo = adjective
fs = preposition
ao = adverb
eo = qualifying adverb

??
nhm = verb infinitive indicator
gr = definitive article
uh = exclamation

??
st = conjunction
stt = connective conjunction

PRP
fn = pronoun
pfn = personal pronoun
abfn = reflexive pronoun

NNP
person = person name
sérnafn = proper name
entity = proper name of reconized named entity
fyrirtæki = company name
gata = street name

CD
to = number word
töl = number word
"""