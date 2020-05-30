import json

from flask import Flask
from flask import request
from flask_cors import CORS

import nltk
from nltk.tokenize.punkt import PunktToken
from nltk.corpus import wordnet

from py2neo import Graph

from MLUtils import MLUtils
from NLPUtils import NLPUtils
from config import *

app = Flask(__name__)
CORS(app)
graph = Graph("bolt://localhost:7687", auth=("TeaNLP", "teanlp"))


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
    if request is not None and request.json is not None:
        request_json = json.loads(request.json, strict=False)
        if 'text' in request_json:

            text = request_json['text'].replace("\n", " " + NEWLINE + " ")

            tokenizer = None
            for s, sentence in enumerate(NLPUtils.sentencize(text)):
                tokens, tokenizer = NLPUtils.tokenize(sentence, tokenizer)
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
                    features[LEMMA] = NLPUtils.lemmatize(token)
                    features[STEM] = NLPUtils.stemize(token)
                    features[IS_STOP] = NLPUtils.is_stop_word(token)
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

                    if not features[IS_STOP]:
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

# 1) Receive the token from FRONT  -> features, BERT embedding
# 2) Receive the sentence from front -> BERT sentence embedding
# 3) Receive the text from front -> BoW, mean value of embeddings of the BoW

def save():
    text = "Hi, my name is Raul, I'm from Spain. I would like to know how are you. Or not..."
    we, se, te = MLUtils.get_embeddings(text, tok_num=4)
    print(we.shape)
    print(se.shape)
    print(te.shape)

if __name__ == "__main__":
    save()
