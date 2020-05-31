from nltk.tokenize.punkt import PunktToken

from NLPUtils import NLPUtils
from domain.tessaurus import Tessaurus


class LinguisticFeatures:
    def __init__(self, orth, pos, offset):
        tok = PunktToken(orth)
        self.orth = orth
        self.lower = orth.lower()
        self.pos = pos
        self.offset = offset
        self.lemma = NLPUtils.lemmatize(self.lower)
        self.stem = NLPUtils.stemize(self.lower)
        self.is_stop = NLPUtils.is_stop_word(self.lower)
        self.is_stop = orth.lower() in NLPUtils.instance.english_vocab
        self.is_alfa = tok.is_alpha
        self.is_num = tok.is_number
        self.is_punct = not tok.is_non_punct
        self.tessaurus = Tessaurus(self.orth, self.pos, self.is_stop)
        self.meaningful_embedding = not self.is_num and not self.is_punct and not self.is_stop

    @property
    def orth(self):
        return self.orth

    @orth.setter
    def orth(self, value):
        self.orth = value

    @property
    def lower(self):
        return self.lower

    @lower.setter
    def lower(self, value):
        self.lower = value

    @property
    def pos(self):
        return self.pos

    @pos.setter
    def pos(self, value):
        self.pos = value

    @property
    def offset(self):
        return self.offset

    @offset.setter
    def offset(self, value):
        self.offset = value

    @property
    def start_lower(self):
        return self.start_lower

    @start_lower.setter
    def start_lower(self, value):
        self.start_lower = value

    @property
    def lemma(self):
        return self.lemma

    @lemma.setter
    def lemma(self, value):
        self.lemma = value

    @property
    def stem(self):
        return self.stem

    @stem.setter
    def stem(self, value):
        self.lemma = value

    @property
    def is_stop(self):
        return self.is_stop

    @is_stop.setter
    def is_stop(self, value):
        self.is_stop = value

    @property
    def is_alpha(self):
        return self.is_alpha

    @is_alpha.setter
    def is_alpha(self, value):
        self.is_alpha = value

    @property
    def is_num(self):
        return self.is_num

    @is_num.setter
    def is_num(self, value):
        self.is_num = value

    @property
    def is_punct(self):
        return self.is_punct

    @is_punct.setter
    def is_punct(self, value):
        self.is_punct = value

    @property
    def tessaurus(self):
        return self.tessaurus

    @tessaurus.setter
    def tessaurus(self, value):
        self.tessaurus = value

    @property
    def meaningful_embedding(self):
        return self.meaningful_embedding

    @meaningful_embedding.setter
    def meaningful_embedding(self, value):
        self.meaningful_embedding = value


