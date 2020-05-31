import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelWithLMHead

from NLPUtils import NLPUtils
from config import NEWLINE


class MLUtils:
    EMBEDDINGS_NAME = 'distilbert-base-cased'
    SENTENCE_EMBEDDINGS_NAME = 'bert-base-nli-mean-tokens'
    SUBWORD_MARK = '##'

    instance = None

    class __MLUtils:
        def __init__(self):
            # This forces download if not done
            self.tokenizer = AutoTokenizer.from_pretrained(MLUtils.EMBEDDINGS_NAME)
            self.model = AutoModelWithLMHead.from_pretrained(MLUtils.EMBEDDINGS_NAME)
            self.sentence_model = SentenceTransformer(MLUtils.SENTENCE_EMBEDDINGS_NAME)

    def __init__(self):
        if not MLUtils.instance:
            MLUtils.instance = MLUtils.__MLUtils()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    @staticmethod
    def subword_tokenize(text):
        original_tokens = MLUtils.instance.tokenizer.tokenize(text)
        lower_tokens = MLUtils.instance.tokenizer.tokenize(text.lower())
        merged_original_tokens = []
        merged_lower_tokens = []

        last_tok = 0
        for i, tok in enumerate(lower_tokens):
            if MLUtils.SUBWORD_MARK in tok and last_tok > 0 and 'text' in merged_lower_tokens[last_tok-1] and 'num' in \
                    merged_lower_tokens[last_tok - 1]:
                merged_lower_tokens[last_tok - 1]['text'] += tok
                merged_lower_tokens[last_tok - 1]['num'].append(i)
            else:
                merged_lower_tokens.append({'text': tok, 'num': [i]})
                last_tok += 1

        last_tok = 0
        for i, tok in enumerate(original_tokens):
            if MLUtils.SUBWORD_MARK in tok and last_tok > 0 and 'text' in merged_original_tokens[last_tok - 1] \
                    and 'num' in merged_original_tokens[last_tok - 1]:
                merged_original_tokens[last_tok - 1]['text'] += tok
                merged_original_tokens[last_tok - 1]['num'].append(i)
            else:
                merged_original_tokens.append({'text': tok, 'num': [i]})
                last_tok += 1

        return merged_original_tokens, merged_lower_tokens, NLPUtils.truecase(merged_original_tokens,
                                                                              merged_lower_tokens)

    @staticmethod
    def tokenize(text):
        return MLUtils.instance.tokenizer.tokenize(text)

    @staticmethod
    def subwords_to_words(merged_tokens):
        return [x['text'].replace(MLUtils.SUBWORD_MARK, '') for x in merged_tokens]

    @staticmethod
    def get_bert_text_embeddings(text):
        # we = PyTorch tensor of (1, N, 768), where N is num of tokens
        # we[0].shape[2] = 768 for BERT

        we = MLUtils.get_bert_word_embeddings(text)
        print(we.shape)
        embeddings_dim = we.shape[2]
        print("Dim: {}".format(embeddings_dim))
        tok_num = we.shape[1]
        print("Tokens: {}".format(tok_num))

        tensor_acc = torch.tensor(np.zeros(embeddings_dim))

        for i in range(tok_num):
            tensor_acc += we[0][i]

        mean = tensor_acc / tok_num

        return mean

    @staticmethod
    def get_bert_word_embeddings(sentence):

        embeddings = torch.tensor(MLUtils.instance.tokenizer.encode(sentence, add_special_tokens=False)).\
            unsqueeze(0)

        return MLUtils.instance.model(embeddings)[0]

    @staticmethod
    def get_bert_sentence_embeddings(sentence):
        # This returns an array of tensors
        sentence_embeddings_numpy = MLUtils.instance.sentence_model.encode(sentence)
        return torch.tensor(sentence_embeddings_numpy)

    """
    def get_embeddings(self, text, original_toks, lower_toks, original_tok_num, lower_tok_num):

        # Word Embeddings
        # hidden_layer, we is a PyTorch tensor of (1, N, 768), where N is num of tokens
        # [0] because is last layer
        we_lower = self.get_bert_word_embeddings(lower_toks)[0]
        we_lower_lemmas = self.get_bert_word_embeddings(lower_toks, lemmatize=True)[0]
        we_original = self.get_bert_word_embeddings(original_toks)[0]

        if original_tok_num >= we_original.shape[1]:
            print("Parameter 'tok_num' bigger than number of tokens. Returning None")
            we_original = None

        if lower_tok_num >= we_lower.shape[1]:
            print("Parameter 'tok_num' bigger than number of tokens. Returning None")
            we_lower = None

        if lower_tok_num >= we_lower_lemmas.shape[1]:
            print("Parameter 'tok_num' bigger than number of tokens. Returning None")
            we_lower_lemmas = None

        # Sentence Embeddings
        se = self.get_bert_sentence_embeddings(text)
        # print("Sentence embeddings tensors:")
        # print(se)

        # Text embeddings: Mean of the word embeddings.
        we_lemmas = self.get_bert_word_embeddings(text, lemmatize=True)[0]
        te = MLUtils.get_bert_text_embeddings(we_lemmas)
        # print("Text embeddings tensors:")
        # print(te)
        # print(te.shape)

        return we[0][tok_num], se[sentence_num], te"""

    @staticmethod
    def token_prediction(masked_sentence, original_word):

        input = MLUtils.instance.tokenizer.encode(masked_sentence, return_tensors="pt")
        mask_token_index = torch.where(input == MLUtils.instance.tokenizer.mask_token_id)[1]

        token_logits = MLUtils.instance.model(input)[0]
        mask_token_logits = token_logits[0, mask_token_index, :]

        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

        decoded_top_5_tokens = [MLUtils.instance.tokenizer.decode([token]) for token in top_5_tokens]
        result = []
        for token in decoded_top_5_tokens:
            if token in set(NLPUtils.instance.english_vocab) and len(token) > 2 and original_word[0].isupper() == \
                    token[0].isupper():
                result.append(token)

        return result
        #for token in top_5_tokens:
        #    print(masked_sentence.replace(MLUtils.instance.tokenizer.mask_token, MLUtils.instance.tokenizer.decode([token])))

