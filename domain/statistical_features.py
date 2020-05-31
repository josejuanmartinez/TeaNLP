import copy

from MLUtils import MLUtils


class StatisticalFeatures:
    def __init__(self, original_subwords, lower_subwords, se, te, token):
        self.bert_subwords_original = original_subwords
        self.bert_subwords_lower = lower_subwords
        self.text_embedding = te
        self.sentence_embedding = se
        self.meaningful_embedding = token.linguistic_features.meaningful_embedding
        self.similar_words = self.predict(token)

    @property
    def bert_subwords_original(self):
        return self.bert_subwords_original

    @bert_subwords_original.setter
    def bert_subwords_original(self, value):
        self.bert_subwords_original = value

    @property
    def bert_subwords_lower(self):
        return self.bert_subwords_lower

    @bert_subwords_lower.setter
    def bert_subwords_lower(self, value):
        self.bert_subwords_lower = value

    @property
    def sentence_embedding(self):
        return self.sentence_embedding

    @sentence_embedding.setter
    def sentence_embedding(self, value):
        self.sentence_embedding = value

    @property
    def similar_words(self):
        return self.similar_words

    @similar_words.setter
    def similar_words(self, value):
        self.similar_words = value

    @property
    def meaningful_embedding(self):
        return self.meaningful_embedding

    @meaningful_embedding.setter
    def meaningful_embedding(self, value):
        self.meaningful_embedding = value

    def predict(self, token):
        predictions = []
        if self.meaningful_embedding:
            masked_tokens = copy.deepcopy(token.original_clean_tokens)
            masked_tokens[token.linguistic_features.order_in_sentence] = MLUtils.instance.tokenizer.mask_token
            masked_sentence = " ".join(masked_tokens)
            predictions = MLUtils.token_prediction(masked_sentence, token)
        return predictions
