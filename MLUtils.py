import copy

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForTokenClassification, pipeline

from NLPUtils import NLPUtils


class MLUtils:
    EMBEDDINGS_NAME = 'distilbert-base-cased'
    NER_EMBEDDINGS_NAME = 'dbmdz/bert-large-cased-finetuned-conll03-english'
    SENTENCE_EMBEDDINGS_NAME = 'bert-base-nli-mean-tokens'
    SUBWORD_MARK = '##'
    NOENT = 'O-NOENT'

    instance = None

    class __MLUtils:
        def __init__(self):
            # This forces download if not done
            self.tokenizer = AutoTokenizer.from_pretrained(MLUtils.EMBEDDINGS_NAME)
            self.model = AutoModelWithLMHead.from_pretrained(MLUtils.EMBEDDINGS_NAME)
            self.ner = AutoModelForTokenClassification.from_pretrained(MLUtils.NER_EMBEDDINGS_NAME)
            self.sentence_model = SentenceTransformer(MLUtils.SENTENCE_EMBEDDINGS_NAME)

    def __init__(self):
        if not MLUtils.instance:
            MLUtils.instance = MLUtils.__MLUtils()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    @staticmethod
    def merge_subwords(tokens, ner_tokens=None):
        last_tok = 0
        result = []
        ner_positions = []
        if ner_tokens is not None and len(ner_tokens) > 0:
            ner_positions = [x['index']-1 for x in ner_tokens]
        for i, tok in enumerate(tokens):
            if MLUtils.SUBWORD_MARK in tok and last_tok > 0 and 'text' in result[last_tok-1] and 'num' in \
                    result[last_tok - 1]:
                result[last_tok - 1]['text'] += tok
                result[last_tok - 1]['num'].append(i)
                if ner_tokens is not None and i in ner_positions:
                    result[last_tok - 1]['ner'].append(ner_tokens[ner_positions.index(i)]['entity'])
                else:
                    result[last_tok - 1]['ner'].append(MLUtils.NOENT)
            else:
                result.append({'text': tok, 'num': [i], 'ner': [MLUtils.NOENT]})
                if ner_tokens is not None and i in ner_positions:
                    result[len(result)-1]['ner'] = [ner_tokens[ner_positions.index(i)]['entity']]
                last_tok += 1
        return result

    @staticmethod
    def tokenize(text):
        return MLUtils.instance.tokenizer.tokenize(text)

    @staticmethod
    def subwords_to_words(merged_tokens):
        return [x.replace(MLUtils.SUBWORD_MARK, '') for x in merged_tokens]

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

    @staticmethod
    def predict(token):
        """
        Predicts based on the merged word in the original sentence, not in the subword
        Args:
            token: Token object using merged original word

        Returns:
            A list of word predictions (semantic synonyms)
        """
        predictions = []
        if token.linguistic_features.meaningful_embedding:
            masked_tokens = copy.deepcopy(token.original_sentence_tokens)
            masked_tokens[token.linguistic_features.offset.order_in_sentence] = MLUtils.instance.tokenizer.mask_token
            masked_sentence = " ".join(masked_tokens)
            predictions = MLUtils.token_prediction(masked_sentence, token)
        return predictions

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
            if token in set(NLPUtils.instance.english_vocab) and len(token) > 2 and not original_word.\
                linguistic_features.orth[0].isupper() == token[0].isupper():
                result.append(token)

        return result
        #for token in top_5_tokens:
        #    print(masked_sentence.replace(MLUtils.instance.tokenizer.mask_token, MLUtils.instance.tokenizer.decode([token])))

    @staticmethod
    def ner(text, grouped_entities=False):
        nlp = pipeline('ner', model=MLUtils.instance.ner, tokenizer=MLUtils.instance.tokenizer, grouped_entities=grouped_entities)
        return nlp(text)
