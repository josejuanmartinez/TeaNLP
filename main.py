import re

from flask import Flask
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktToken
from py2neo import Graph
from flask import request
import nltk
from nltk import TweetTokenizer
import json
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet

app = Flask(__name__)
CORS(app)
graph = Graph("bolt://localhost:7687", auth=("TeaNLP", "teanlp"))

NEWLINE = '_NEWLINE_'

ORDER = "ORDER"
ORTH = "ORTH"
LOWER = "LOWER"
POS = "POS"
LEMMA = "LEMMA"
STEM = "STEM"
IS_PUNCT = "IS_PUNCT"
IS_STOP = "IS_STOP"
IS_NUM = "IS_NUM"
IS_ALPHA = "IS_ALPHA"
HYPERNYM = "HYPERNYM"
SYNONYM = "SYNONYM"
ANTONYM = "ANTONYM"


def check_prerequisites():
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


@app.route("/")
def hello():
    graph.run("Match () Return 1 Limit 1")
    return 'Hello, Amparo!'


@app.route("/preprocess", methods=['POST'])
def preprocess():
    check_prerequisites()
    preprocessed_tokens = []
    tokens = []
    if request is not None and request.json is not None:
        request_json = json.loads(request.json, strict=False)
        if 'text' in request_json:
            text = request_json['text'].replace("\n", " " + NEWLINE + " ")
            tokens = TweetTokenizer().tokenize(text)
            for i in range(0, len(tokens)):
                if tokens[i] == NEWLINE:
                    tokens[i] = '\n'

    pos = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words('english'))

    for i, token in enumerate(tokens):
        tok = PunktToken(token)

        features = dict()
        features[ORDER] = str(i)
        features[ORTH] = token
        features[LOWER] = token.lower()
        features[POS] = pos[i][1]
        features[LEMMA] = lemmatizer.lemmatize(token)
        features[STEM] = stemmer.stem(token)
        features[IS_STOP] = token in stop_words
        features[IS_ALPHA] = tok.is_alpha is not None
        features[IS_NUM] = tok.is_number
        features[IS_PUNCT] = not tok.is_non_punct

        synonyms = set()
        antonyms = set()
        hypernyms = set()
        wordnet_pos = ''  # n,v,a,r

        if features[POS].startswith('JJ'):
            wordnet_pos = 'a'
        elif features[POS].startswith('NN'):
            wordnet_pos = 'n'
        elif features[POS].startswith('VB'):
            wordnet_pos = 'v'
        elif features[POS].startswith('RB'):
            wordnet_pos = 'r'

        if not features[IS_STOP] and not features[IS_NUM] and not features[IS_PUNCT]:
            for syn in wordnet.synsets(token, pos=wordnet_pos):
                for l in syn.lemmas():
                    synonym = l.name().replace("_", " ").replace("-", " ").lower()
                    if synonym != token.lower():
                        synonyms.add(synonym)
                    if l.antonyms():
                        antonyms.add(l.antonyms()[0].name().replace("_", " ").replace("-", " ").lower())

                for hypernym in syn.hypernyms():
                    for lemma in hypernym.lemmas():
                        hypernyms.add(lemma.name().replace("_", " ").replace("-", " ").lower())

        features[HYPERNYM] = list(hypernyms)
        features[SYNONYM] = list(synonyms)
        features[ANTONYM] = list(antonyms)

        preprocessed_tokens.append(features)

    print(preprocessed_tokens)
    return {'result': preprocessed_tokens}

