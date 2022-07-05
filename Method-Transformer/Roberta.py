from transformers import RobertaTokenizer, RobertaModel
from transformers_domain_adaptation import DataSelector
from transformers_domain_adaptation import VocabAugmentor

model_card = 'roberta-base'

# Load model and tokenizer
model = RobertaModel.from_pretrained(model_card)
tokenizer = RobertaTokenizer.from_pretrained(model_card)

# Reference: https://colab.research.google.com/github/georgianpartners/Transformers-Domain-Adaptation/blob/master/notebooks/GuideToTransformersDomainAdaptation.ipynb#scrollTo=J-fwmTN74i3f
selector = DataSelector(
    keep = 0.5,
    tokenizer = tokenizer,
    similarity_metrics=['euclidean']
    diversity_metrics = [
        'type_token_ratio',
        'entropy',
    ],
)

