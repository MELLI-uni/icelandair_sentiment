import pickle
from itertools import islice
from translate import Translator

dictionaries  = []

# Make English emoji dictionary from emoticon and emoji file
# Make Icelandic emoji dictionary by translating English into Icelandic (with translate api)
translator = Translator(to_lang="is")
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
dictionaries.append(flight_list)

# Make icelandic stopword list from stopword lexicon
isk_stop = []
with open('../../lexicons/isk_stop.txt', encoding='utf-8') as f:
    for line in f:
        isk_stop.append(line.strip())
dictionaries.append(isk_stop)

# with open('dictionaries.pickle', 'rb') as handle:
#     tmp = pickle.load(handle)

# dictionaries.append(tmp[0])
# dictionaries.append(tmp[1])
# dictionaries.append(tmp[2])
# dictionaries.append(tmp[3])

isk_modal = []
with open('../../lexicons/isk_modal.txt', encoding='utf-8') as f:
    for line in f:
        isk_modal.append(line.strip())
dictionaries.append(isk_modal)

# Dictionary of English degree adverbs
ENG_ADV = {'a bit':'DEC', 'adequately':'DEC', 'almost':'DEC', 'barely':'DEC', 'fairly':'DEC', 'hardly':'DEC', 'just enough':'DEC', 'kind of':'DEC', 'kinda':'DEC', 
            'kindof':'DEC', 'kind-of':'DEC', 'less':'DEC', 'little':'DEC', 'marginal':'DEC', 'marginally':'DEC', 'moderately':'DEC', 'modest':'DEC', 'nearly':'DEC', 
            'occasional':'DEC', 'occasionally':'DEC', 'partly':'DEC', 'scarce':'DEC', 'scarcely':'DEC', 'seldom':'DEC', 'slight':'DEC', 'slightly':'DEC', 
            'somewhat':'DEC', 'sort of':'DEC', 'sorta':'DEC', 'sortof':'DEC', 'sort-of':'DEC', 'sufficiently':'DEC', 

            '100 percent':'INC', '100-percent':'INC', '100%':'INC', 'a lot':'INC', 'alot':'INC', 'absolutely':'INC', 'amazingly':'INC', 'awfully':'INC', 'clearly':'INC',
            'completely':'INC', 'considerable':'INC', 'considerably':'INC', 'decidedly':'INC', 'deeply':'INC', 'enormous':'INC', 'enormously':'INC', 'entirely':'INC', 
            'especially':'INC', 'exceedingly':'INC', 'exceptional':'INC', 'exceptionally':'INC', 'excessively':'INC', 'extensively':'INC', 'extra':'INC', 'extreme':'INC', 
            'extremely':'INC', 'fabulously':'INC', 'fantastically':'INC', 'fully':'INC', 'greatly':'INC', 'highly':'INC', 'hugely':'INC', 'incredible':'INC', 
            'incredibly':'INC', 'intensely':'INC', 'largely':'INC', 'major':'INC', 'majorly':'INC', 'more':'INC', 'most':'INC', 'much':'INC', 'noticeably':'INC', 
            'particularly':'INC', 'perfectly':'INC', 'positively':'INC', 'pretty':'INC', 'purely':'INC', 'quite':'INC', 'really':'INC', 'reasonably':'INC', 
            'remarkably':'INC', 'so':'INC', 'strikingly':'INC', 'strongly':'INC', 'substantially':'INC', 'thoroughly':'INC', 'too':'INC', 'total':'INC', 'totally':'INC', 
            'tremendous':'INC', 'tremendously':'INC', 'truly':'INC', 'uber':'INC', 'unbelievably':'INC', 'unusually':'INC', 'usually':'INC', 'utter':'INC', 
            'utterly':'INC', 'very':'INC', 'well':'INC'
            }
dictionaries.append(ENG_ADV)

# Dictionary of icelandic degree adverbs
ISK_ADV = {'að hluta':'DEC', 'af skornum skammti':'DEC', 'bara nóg':'DEC', 'fullnægjandi':'DEC', 'hóflega':'DEC', 'hóflegur':'DEC', 'hófsamur':'DEC', 
            'jaðar':'DEC', 'lítillega':'DEC', 'lítilsháttar':'DEC', 'litla':'DEC', 'minna':'DEC', 'nægilega':'DEC', 'næstum':'DEC', 'næstum því':'DEC', 
            'nokkuð':'DEC', 'örlítið':'DEC', 'sjaldan':'DEC', 'stöku sinnum':'DEC', 'stundum':'DEC', 'svoleiðis':'DEC', 'svolítið':'DEC', 'svona':'DEC', 
            'varla':'DEC',
            
            '100 prósent':'INC', '100-prósent':'INC', '100%':'INC', '100 %':'INC', 'að fullu':'INC', 'að mestu leyti':'INC', 'að miklu leyti':'INC', 
            'að öllu leyti':'INC', 'aðallega':'INC', 'afbrigðilegur':'INC', 'afskaplega':'INC', 'ákaflega':'INC', 'ákveðið':'INC', 'alger':'INC', 'algerlega':'INC', 
            'alveg':'INC', 'auka':'INC', 'djúpt':'INC', 'eindregið':'INC', 'eingöngu':'INC', 'frábærlega':'INC', 'frekar':'INC', 'fullkomlega':'INC', 'gersamlega':'INC', 
            'greinilega':'INC', 'gríðarlega':'INC', 'jæja':'INC', 'jákvætt':'INC', 'líka':'INC', 'með öllu':'INC', 'meira':'INC', 'meiriháttar':'INC', 'merkilega':'INC',
            'merkjanlega':'INC', 'mest':'INC', 'mikið':'INC', 'mikill':'INC', 'mjög':'INC', 'öfgafullt':'INC', 'óhóflega':'INC', 'ótrúlega':'INC', 'ótrúlegur':'INC', 
            'óvenju':'INC', 'óvenjulegur':'INC', 'rækilega':'INC', 'raunverulega':'INC', 'sæmilega':'INC', 'sætur':'INC', 'samtals':'INC', 'sannarlega':'INC', 
            'sérstaklega':'INC', 'sláandi':'INC', 'stórkostlega':'INC', 'stórkostlegt':'INC', 'svo':'INC', 'talsverður':'INC', 'talsvert':'INC', 
            'undantekningarlaust':'INC', 'vel':'INC', 'venjulega':'INC', 'verulega':'INC', 'virkilega':'INC'
            }
dictionaries.append(ISK_ADV)

# Expanding dictionary
# with open('dictionaries.pickle', 'rb') as handle:
#     temp = pickle.load(handle)

# dictionaries.append(temp[0])
# dictionaries.append(temp[1])
# dictionaries.append(temp[2])
# dictionaries.append(temp[3])
# dictionaries.append(temp[4])
# dictionaries.append(temp[5])

# Store data (serialize)
with open('dictionaries.pickle', 'wb') as handle:
    pickle.dump(dictionaries, handle, protocol=pickle.HIGHEST_PROTOCOL)