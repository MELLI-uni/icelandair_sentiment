from tqdm import tqdm
import torch
import torch.nn. as nn
import torch.nn.functional as F

from torchtext import data, datasets
import random
import timeit

Seed = 77
random.seed(Seed)
torch.manual_seed(Seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=True, batch_first=True)

trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

vars(trainset.examples[0])