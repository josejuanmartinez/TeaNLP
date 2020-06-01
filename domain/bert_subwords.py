import json

from MLUtils import MLUtils


class BertSubwords:
    def __init__(self, text_num_dict, we):
        self._subwords = text_num_dict['text']
        self._root = self.subwords.split(MLUtils.SUBWORD_MARK)[0]
        self._length = len(text_num_dict['num'])
        self._is_subwords = self.length > 1
        self._embedding = we

    def to_json(self, printable_embeddings=True):
        return {"subwords": self._subwords, "root": self._root, "length": self._length,
                    "is_subwords": self._is_subwords, "embedding": self.embedding[-10:] if printable_embeddings else self._embedding}

    @property
    def subwords(self):
        return self._subwords

    @subwords.setter
    def subwords(self, value):
        self._subwords = value

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        self._root = value

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value

    @property
    def is_subwords(self):
        return self._is_subwords

    @is_subwords.setter
    def is_subwords(self, value):
        self._is_subwords = value

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, value):
        self._embedding = value

