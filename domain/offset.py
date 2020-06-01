import json


class Offset:
    def __init__(self, sentence_num, start_original, start_merged, start_lower, order_in_sentence):
        self._sentence_num = sentence_num
        self._start_original = start_original
        self._start_merged = start_merged
        self._start_lower = start_lower
        self._order_in_sentence = order_in_sentence

    def to_json(self):
        return {"sentence_num": self._sentence_num, "start_original": self._start_original,
                    "start_merged": self._start_merged, "start_lower": self._start_lower,
                    "order_in_sentence": self._order_in_sentence}

    @property
    def sentence_num(self):
        return self._sentence_num

    @sentence_num.setter
    def sentence_num(self, value):
        self._sentence_num = value

    @property
    def start_original(self):
        return self._start_original

    @start_original.setter
    def start_original(self, value):
        self._start_original = value

    @property
    def start_merged(self):
        return self._start_merged

    @start_merged.setter
    def start_merged(self, value):
        self._start_merged = value

    @property
    def start_lower(self):
        return self._start_lower

    @start_lower.setter
    def start_lower(self, value):
        self._start_lower = value

    @property
    def order_in_sentence(self):
        return self._order_in_sentence

    @order_in_sentence.setter
    def order_in_sentence(self, value):
        self._order_in_sentence = value
