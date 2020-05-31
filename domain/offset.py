class Offset:
    def __init__(self, sentence_num, start_original, start_merged, start_lower, order_in_sentence):
        self.sentence_num = sentence_num
        self.start_original = start_original
        self.start_merged = start_merged
        self.start_lower = start_lower
        self.order_in_sentence = order_in_sentence

    @property
    def sentence_num(self):
        return self.sentence_num

    @sentence_num.setter
    def sentence_num(self, value):
        self.sentence_num = value

    @property
    def start_original(self):
        return self.start_original

    @start_original.setter
    def start_original(self, value):
        self.start_original = value

    @property
    def start_merged(self):
        return self.start_merged

    @start_merged.setter
    def start_merged(self, value):
        self.start_merged = value

    @property
    def start_lower(self):
        return self.start_lower

    @start_lower.setter
    def start_lower(self, value):
        self.start_lower = value

    @property
    def order_in_sentence(self):
        return self.order_in_sentence

    @order_in_sentence.setter
    def order_in_sentence(self, value):
        self.order_in_sentence = value
