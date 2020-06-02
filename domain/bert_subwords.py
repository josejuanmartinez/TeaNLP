from MLUtils import MLUtils


class BertSubwords:
    def __init__(self, text_num_dict, we):
        self._subwords = text_num_dict['text']
        self._root = self.subwords.split(MLUtils.SUBWORD_MARK)[0]
        self._length = len(text_num_dict['num'])
        self._is_subwords = self.length > 1
        self._meaningful_embedding = self._length < 3 and len(self._root) > 2 and len(self._root) > \
                                     len(self.subwords.split(MLUtils.SUBWORD_MARK)[1:])
        self._embedding = we if self._meaningful_embedding else None

    def to_json(self, printable_embeddings=True):
        return {"subwords": self._subwords, "root": self._root, "length": self._length,
                "is_subwords": self._is_subwords, "meaningful_embedding": self._meaningful_embedding,
                "embedding": self.embedding[-10:] if printable_embeddings and self._embedding is not None else None}

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

    @property
    def meaningful_embedding(self):
        return self._meaningful_embedding

    @meaningful_embedding.setter
    def meaningful_embedding(self, value):
        self._meaningful_embedding = value

