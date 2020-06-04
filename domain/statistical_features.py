from MLUtils import MLUtils


class StatisticalFeatures:
    def __init__(self, original_subwords, lower_subwords, se, te, token):
        self._bert_subwords_original = original_subwords
        self._bert_subwords_lower = lower_subwords
        self._text_embedding = te
        self._sentence_embedding = se
        self._similar_words = MLUtils.predict(token)

    def to_json(self, printable_embeddings=True):
        return {"bert_subwords_original": self._bert_subwords_original.to_json(printable_embeddings),
                "bert_subwords_lower": self._bert_subwords_lower.to_json(printable_embeddings),
                "text_embedding": self._text_embedding[-10:] if printable_embeddings else self._text_embedding,
                "sentence_embedding": self._sentence_embedding[-10:] if printable_embeddings
                else self._sentence_embedding, "similar_words": self._similar_words}

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
