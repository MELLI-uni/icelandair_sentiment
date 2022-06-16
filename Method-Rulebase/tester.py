import gensim.downloader
from nltk.corpus import words
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from textblob import TextBlob
from textblob.en import Spelling



# def find_similar(word, lexicon):
#     similar_tokens = [x[0] for x in glove_vectors.most_similar(word)]

#     for token in similar_tokens:
#         if token in lexicon:
#             return token

#     return word

# def find_in_lexicon(tokens, lexicon):
#     score = []

#     for i in tokens:
#         if i in lexicon:
#             print(i, "pass")
#             continue

#         similar = find_similar(i, lexicon)
#         if similar in lexicon:
#             print(similar, "pass")
#             continue

#         print("0")

# glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
# # whitelist = list(wn.words()) + words.words()

# lexicon = ['food', 'lactose', 'happy', 'caffeine']
# tokens = ['happy', 'gluten', 'food']

# find_in_lexicon(tokens, lexicon)

english_words = list(wn.words()) + words.words() + ['covid']
blank_list = []

def check_word(word):
    if word in english_words:
        return True

    return False

def divide_word(word):
    word1 = word[:len(word)//2]
    word2 = word[len(word)//2:]

    return word1, word2

# spell = SpellChecker(english_words)

spell = SpellChecker()
temp = spell.correction('foodsdrinksmedicine')

if check_word(temp) == True:
    blank_list.apend(temp)

print(check_word(temp))

# glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
# if temp in glove_vectors:
#     print(glove_vectors.most_similar(temp))
# else:
#     print("not exist")

# if score > 0.8:
#     print(word)
# else:
#     print("bad")



# word = 'menubrochure'
# if word not in whitelist:
#     print(spell.correction(word))
# else:
#     print(word)

# from spacy.lang.en import English

# nlp = English()

# def spell_correct_eng(input):
#     tokens = word_tokenize(input)

#     corrected_tokens = []

#     for token in tokens:
#         if token not in whitelist:
#             corrected_tokens.append(spell.correction(token))
#             continue
#         corrected_tokens.append(token)

#     return " ".join(t for t in corrected_tokens)

# text = "I don't like the place"

# print(spell_correct_eng(text))