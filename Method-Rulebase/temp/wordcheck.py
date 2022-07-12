from gensim.models import Word2Vec

model = Word2Vec.load('icelandic_word2vec.model')

print(model.wv.most_similar('ykkur'))
