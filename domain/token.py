from domain.linguistic_features import LinguisticFeatures
from domain.statistical_features import StatisticalFeatures


class Token:
    def __init__(self, orth, pos, offset, original_subwords, lower_subwords, se, te, original_sentence_tokens):
        self.original_sentence_tokens = original_sentence_tokens
        self.linguistic_features = LinguisticFeatures(orth, pos, offset)
        self.statistical_features = StatisticalFeatures(original_subwords, lower_subwords, se, te, self)

    def __repr__(self):
        return "{'original_sentence_tokens': {}, 'linguistic_features': {}, 'statistical_features': {}}".format(
            self.original_sentence_tokens, repr(self.linguistic_features), repr(self.statistical_features))

    @property
    def linguistic_features(self):
        return self.linguistic_features

    @linguistic_features.setter
    def linguistic_features(self, value):
        self.linguistic_features = value

    @property
    def statistical_features(self):
        return self.statistical_features

    @statistical_features.setter
    def statistical_features(self, value):
        self.statistical_features = value
