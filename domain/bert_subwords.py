from MLUtils import MLUtils


class BertSubwords:
    def __init__(self, text_num_dict, we):
        self.subwords = text_num_dict['text']
        self.root = self.subwords.split(MLUtils.SUBWORD_MARK)[0]
        self.length = len(text_num_dict['num'])
        self.is_subwords = self.length > 1
        self.embedding = None

    @property
    def subwords(self):
        return self.subwords

    @subwords.setter
    def subwords(self, value):
        self.subwords = value

    @property
    def root(self):
        return self.root

    @root.setter
    def root(self, value):
        self.root = value

    @property
    def length(self):
        return self.length

    @length.setter
    def length(self, value):
        self.length = value

    @property
    def is_subwords(self):
        return self.is_subwords

    @is_subwords.setter
    def is_subwords(self, value):
        self.is_subwords = value

    @property
    def embedding(self):
        return self.embedding

    @embedding.setter
    def embedding(self, value):
        self.embedding = value

