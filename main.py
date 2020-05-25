import json

from flask import Flask
from flask import request
from flask_cors import CORS

import nltk
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktToken
from nltk import TweetTokenizer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet

from py2neo import Graph

import torch
from transformers import *

app = Flask(__name__)
CORS(app)
graph = Graph("bolt://localhost:7687", auth=("TeaNLP", "teanlp"))

NEWLINE = '_NEWLINE_'

ORDER = "ORDER"
SENTENCE = "SENTENCE"
ORDER_IN_SENTENCE = "ORDER_IN_SENTENCE"
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


def tokenize(text):
    pretrained_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    return tokenizer.tokenize(text)


@app.route("/")
def hello():
    graph.run("Match () Return 1 Limit 1")
    return 'Hello, Amparo!'


@app.route("/preprocess", methods=['POST'])
def preprocess():
    check_prerequisites()
    preprocessed_tokens = []
    if request is not None and request.json is not None:
        request_json = json.loads(request.json, strict=False)
        if 'text' in request_json:

            lemmatizer = WordNetLemmatizer()
            stemmer = SnowballStemmer("english")
            stop_words = set(stopwords.words('english'))

            text = request_json['text'].replace("\n", " " + NEWLINE + " ")
            sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
            sentences = sent_detector.tokenize(text.strip())
            for s, sentence in enumerate(sentences):
                tokens = tokenize(sentence)
                pos = nltk.pos_tag(tokens)
                for t, token in enumerate(tokens):
                    if token == NEWLINE:
                        token = '\n'

                    tok = PunktToken(token)

                    features = dict()
                    features[SENTENCE] = str(s)
                    features[ORDER] = str(s+t)
                    features[ORDER_IN_SENTENCE] = str(t)
                    features[ORTH] = token
                    features[LOWER] = token.lower()
                    features[POS] = pos[t][1]
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


def save():
    text = "Hi, my name is Raul, I'm from Spain"

    lemmatizer = WordNetLemmatizer()

    lemmas = []

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_detector.tokenize(text.strip())
    for s, sentence in enumerate(sentences):
        toks = TweetTokenizer().tokenize(sentence)
        for t, tok in enumerate(toks):
            if tok == NEWLINE:
                tok = '\n'

            lemma = lemmatizer.lemmatize(tok)
            lemmas.append(lemma)

    pretrained_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    model = BertModel.from_pretrained(pretrained_weights)

    input_ids = torch.tensor(tokenizer.encode(' '.join(lemmas), add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    print(outputs[0]) # hidden_layer, PyTorch tensor of (1, N, 768), where N is num of tokens
    print(outputs[0].shape)
    print(outputs[0][0][13])
    print(len(lemmas))
    """
    check_prerequisites()
    spark = sparknlp.start()

    if request is not None and request.json is not None:
        request_json = json.loads(request.json, strict=False)
        if 'graph' in request_json and 'text' in request_json['graph'] and 'ORTH' in request_json['graph']:

            # Lemmas
            lemmatizer.lemmatize(token)

 
    """

if __name__ == "__main__":
    save()
