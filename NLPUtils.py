import nltk
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktToken
from transformers import BertTokenizer

from config import NEWLINE


class NLPUtils:

    @staticmethod
    def is_stop_word(text):
        stop_words = set(stopwords.words('english'))
        tok = PunktToken(text)
        return text in stop_words or tok.is_number or not tok.is_non_punct

    @staticmethod
    def tokenize(text, tokenizer):
        if tokenizer is None:
            pretrained_weights = 'bert-base-uncased'
            tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        return tokenizer.tokenize(text), tokenizer

    @staticmethod
    def sentencize(text):
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        return sent_detector.tokenize(text.strip())

    @staticmethod
    def lemmatize(word):
        lemmatizer = nltk.WordNetLemmatizer()
        return lemmatizer.lemmatize(word)

    @staticmethod
    def stemize(word):
        stemmer = nltk.SnowballStemmer("english")
        return stemmer.stem(word)

    @staticmethod
    def get_bow(text, only_lemmas=False):
        input = []

        tokenizer = None
        for s, sentence in enumerate(NLPUtils.sentencize(text.strip())):
            toks, tokenizer = NLPUtils.tokenize(sentence, tokenizer)
            for t, tok in enumerate(toks):
                if tok == NEWLINE:
                    tok = '\n'

                if only_lemmas:
                    input.append(NLPUtils.lemmatize(tok))
                else:
                    input.append(tok)

        return list(filter(lambda x: not NLPUtils.is_stop_word(x), input))
