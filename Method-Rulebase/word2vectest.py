import gensim.downloader

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')

similar = glove_vectors.most_similar('twitter')

for s in similar:
    print(s[0])
