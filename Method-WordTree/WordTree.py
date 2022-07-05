from gensim.models import Word2Vec
from gensim.models import Phrases

import regex as re
from tqdm import tqdm
from datetime import datetime

# Measure elapsed time to train the model
start = datetime.now()

model = Word2Vec(sentences=common_texts, size=200, window=5, min_count=1, workers=4)
model.save('./Models/wordtree.model')

print(datetime.now() - start)

# Allow to detect phrases longer than one word usiing collocation statistics
bigram_transformer = Phrases(common_texts)
model = Word2Vec(bigram_transformer[common_texts], min_count=1)
