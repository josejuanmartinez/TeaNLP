import copy

from MLUtils import MLUtils


class StatisticalFeatures:
    def __init__(self, original_subwords, lower_subwords, se, te, token):
        self._bert_subwords_original = original_subwords
        self._bert_subwords_lower = lower_subwords
        self._text_embedding = te
        self._sentence_embedding = se
        self._meaningful_embedding = token.linguistic_features.meaningful_embedding
        self._similar_words = self.predict(token)

    def to_json(self, printable_embeddings=True):
        return {"bert_subwords_original": self._bert_subwords_original.to_json(printable_embeddings), "bert_subwords_lower":
            self._bert_subwords_lower.to_json(printable_embeddings), "text_embedding": self._text_embedding[-10:] if printable_embeddings else self._text_embedding, "sentence_embedding":
                        self._sentence_embedding[-10:] if printable_embeddings else self._sentence_embedding, "meaningful_embedding": self._meaningful_embedding,
                    "similar_words": self._similar_words}

    @property
    def bert_subwords_original(self):
        return self._bert_subwords_original

    @bert_subwords_original.setter
    def bert_subwords_original(self, value):
        self._bert_subwords_original = value

    @property
    def bert_subwords_lower(self):
        return self._bert_subwords_lower

    @bert_subwords_lower.setter
    def bert_subwords_lower(self, value):
        self._bert_subwords_lower = value

    @property
    def sentence_embedding(self):
        return self._sentence_embedding

    @sentence_embedding.setter
    def sentence_embedding(self, value):
        self._sentence_embedding = value

    @property
    def similar_words(self):
        return self._similar_words

    @similar_words.setter
    def similar_words(self, value):
        self._similar_words = value

    @property
    def meaningful_embedding(self):
        return self._meaningful_embedding

    @meaningful_embedding.setter
    def meaningful_embedding(self, value):
        self._meaningful_embedding = value

    def predict(self, token):
        predictions = []
        if self.meaningful_embedding:
            masked_tokens = copy.deepcopy(token.original_sentence_tokens)
            masked_tokens[token.linguistic_features.offset.order_in_sentence] = MLUtils.instance.tokenizer.mask_token
            masked_sentence = " ".join(masked_tokens)
            predictions = MLUtils.token_prediction(masked_sentence, token)
        return predictions
