import json

from NLPUtils import NLPUtils


class Tessaurus:
    def __init__(self, orth, pos, is_stop):
        tess = NLPUtils.get_tessaurus(orth, pos, is_stop)
        self._synonyms = tess[0]
        self._antonyms = tess[1]
        self._hypernyms = tess[2]

    def to_json(self):
        return {"synonyms": self._synonyms, "antonyms": self._antonyms,
                     "hypernyms": self._hypernyms}

    @property
    def synonyms(self):
        return self._synonyms

    @synonyms.setter
    def synonyms(self, value):
        self._synonyms = value

    @property
    def antonyms(self):
        return self._antonyms

    @antonyms.setter
    def antonyms(self, value):
        self._antonyms = value

    @property
    def hypernyms(self):
        return self._hypernyms

    @hypernyms.setter
    def hypernyms(self, value):
        self._hypernyms = value
