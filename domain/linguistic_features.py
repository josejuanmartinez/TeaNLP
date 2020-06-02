import json
import re

from nltk.tokenize.punkt import PunktToken

from NLPUtils import NLPUtils
from domain.tessaurus import Tessaurus


class LinguisticFeatures:
    def __init__(self, orth, pos, offset):
        tok = PunktToken(orth)
        self._orth = orth
        self._lower = orth.lower()
        self._pos = pos
        self._offset = offset
        self._lemma = NLPUtils.lemmatize(self.lower)
        self._stem = NLPUtils.stemize(self.lower)
        self._is_stop = NLPUtils.is_stop_word(self.lower)
        # self.is_oov = orth.lower() in NLPUtils.instance.english_vocab
        self._is_alpha = tok.is_alpha is not None
        self._is_num = tok.is_number
        self._is_punct = not tok.is_non_punct
        self._is_space = NLPUtils.is_space(orth)
        self._tessaurus = Tessaurus(self.orth, self.pos, self.is_stop)
        self._meaningful_embedding = not self.is_num and not self.is_punct and not self.is_stop

    def to_json(self):

        return {"orth": self._orth, "lower": self._lower, "pos": self._pos,
                "offset": self._offset.to_json(), "lemma": self._lemma, "stem": self._stem,
                "is_stop": self._is_stop, "is_alpha": self._is_alpha, "is_num": self._is_num,
                "is_punct": self._is_punct, "is_space": self._is_space, "tessaurus": self._tessaurus.to_json(),
                "meaningful_embedding": self._meaningful_embedding}

    @property
    def orth(self):
        return self._orth

    @orth.setter
    def orth(self, value):
        self._orth = value

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, value):
        self._lower = value

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value

    @property
    def lemma(self):
        return self._lemma

    @lemma.setter
    def lemma(self, value):
        self._lemma = value

    @property
    def stem(self):
        return self._stem

    @stem.setter
    def stem(self, value):
        self._stem = value

    @property
    def is_stop(self):
        return self._is_stop

    @is_stop.setter
    def is_stop(self, value):
        self._is_stop = value

    @property
    def is_alpha(self):
        return self._is_alpha

    @is_alpha.setter
    def is_alpha(self, value):
        self._is_alpha = value

    @property
    def is_num(self):
        return self._is_num

    @is_num.setter
    def is_num(self, value):
        self._is_num = value

    @property
    def is_punct(self):
        return self._is_punct

    @is_punct.setter
    def is_punct(self, value):
        self._is_punct = value

    @property
    def is_space(self):
        return self._is_space

    @is_space.setter
    def is_space(self, value):
        self._is_space = value

    @property
    def tessaurus(self):
        return self._tessaurus

    @tessaurus.setter
    def tessaurus(self, value):
        self._tessaurus = value

    @property
    def meaningful_embedding(self):
        return self._meaningful_embedding

    @meaningful_embedding.setter
    def meaningful_embedding(self, value):
        self._meaningful_embedding = value
