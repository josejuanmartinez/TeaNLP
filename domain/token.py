import json

from domain.linguistic_features import LinguisticFeatures
from domain.statistical_features import StatisticalFeatures


class Token:
    def __init__(self, orth, pos, offset, original_subwords, lower_subwords, se, te, original_sentence_tokens):
        self._original_sentence_tokens = original_sentence_tokens
        self._linguistic_features = LinguisticFeatures(orth, pos, offset)
        self._statistical_features = StatisticalFeatures(original_subwords, lower_subwords, se, te, self)

    def to_json(self, printable_embeddings=True):
        return {"original_sentence_tokens": json.dumps(self._original_sentence_tokens), "linguistic_features":
            self._linguistic_features.to_json(), "statistical_features": self._statistical_features.to_json(printable_embeddings)}

    @property
    def original_sentence_tokens(self):
        return self._original_sentence_tokens

    @original_sentence_tokens.setter
    def original_sentence_tokens(self, value):
        self._original_sentence_tokens = value

    @property
    def linguistic_features(self):
        return self._linguistic_features

    @linguistic_features.setter
    def linguistic_features(self, value):
        self._linguistic_features = value

    @property
    def statistical_features(self):
        return self._statistical_features

    @statistical_features.setter
    def statistical_features(self, value):
        self._statistical_features = value
