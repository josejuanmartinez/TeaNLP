import nltk
from nltk.corpus import stopwords, words, wordnet
from nltk.tokenize.punkt import PunktToken


class NLPUtils:

    class __NLPUtils:
        def __init__(self):
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')

            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger')

            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')

            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet')

            try:
                nltk.data.find('corpora/words')
            except LookupError:
                nltk.download('words')

            try:
                nltk.data.find('tokenizers/punkt/english.pickle')
            except LookupError:
                nltk.download('tokenizers/punkt/english.pickle')

            self.english_vocab = set(words.words())
            self.stop_words = set(stopwords.words('english'))
            self.sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    instance = None

    def __init__(self):
        if not NLPUtils.instance:
            NLPUtils.instance = NLPUtils.__NLPUtils()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    @staticmethod
    def is_stop_word(text):
        tok = PunktToken(text)
        return text in NLPUtils.instance.stop_words or tok.is_number or not tok.is_non_punct

    @staticmethod
    def sentencize(text):
        return NLPUtils.instance.sentence_detector.tokenize(text.strip())

    @staticmethod
    def lemmatize(word):
        lemmatizer = nltk.WordNetLemmatizer()
        return lemmatizer.lemmatize(word)

    @staticmethod
    def stemize(word):
        stemmer = nltk.SnowballStemmer("english")
        return stemmer.stem(word)

    @staticmethod
    def truecase(input_toks, output_toks):
        # print("Original {}".format(input_toks))
        # print("Tokenized {}".format(output_toks))
        assert len(input_toks) == len (output_toks)

        result = []
        for i in range(0, len(output_toks)):
            if input_toks[i]['text'].isupper():
                result.append({'text': output_toks[i]['text'].upper(), 'num': output_toks[i]['num']})
            elif input_toks[i]['text'][0].isupper():
                result.append({'text': output_toks[i]['text'][0].upper() + output_toks[i]['text'][1:], 'num': output_toks[i]['num']})
            else:
                result.append({'text': output_toks[i]['text'], 'num': output_toks[i]['num']})
        # print("Truecased {}".format(result))
        return result

    @staticmethod
    def get_tessaurus(token, POS, IS_STOP):

        synonyms = set()
        antonyms = set()
        hypernyms = set()
        wordnet_pos = ''  # n,v,a,r

        if POS.startswith('JJ'):
            wordnet_pos = 'a'
        elif POS.startswith('NN'):
            wordnet_pos = 'n'
        elif POS.startswith('VB'):
            wordnet_pos = 'v'
        elif POS.startswith('RB'):
            wordnet_pos = 'r'

        if not IS_STOP:
            for syn in wordnet.synsets(token, pos=wordnet_pos):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
                    if synonym != token.lower():
                        synonyms.add(synonym)
                    if lemma.antonyms():
                        antonyms.add(lemma.antonyms()[0].name().replace("_", " ").replace("-", " ").lower())

                for hypernym in syn.hypernyms():
                    for lemma in hypernym.lemmas():
                        hypernyms.add(lemma.name().replace("_", " ").replace("-", " ").lower())

        return list(synonyms), list(antonyms), list(hypernyms)


    """
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
    """
