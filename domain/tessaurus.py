from NLPUtils import NLPUtils


class Tessaurus:
    def __init__(self, orth, pos, is_stop):
        tess = NLPUtils.get_tessaurus(orth, pos, is_stop)
        self.synonyms = tess[0]
        self.antonyms = tess[1]
        self.hypernyms = tess[2]

    @property
    def synonyms(self):
        return self.synonyms

    @synonyms.setter
    def synonyms(self, value):
        self.synonyms = value

    @property
    def antonyms(self):
        return self.antonyms

    @antonyms.setter
    def antonyms(self, value):
        self.antonyms = value

    @property
    def hypernyms(self):
        return self.hypernyms

    @hypernyms.setter
    def hypernyms(self, value):
        self.hypernyms = value
