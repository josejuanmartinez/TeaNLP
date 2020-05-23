from flask import Flask
from py2neo import Graph
from flask import request
import nltk
from nltk import TweetTokenizer
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
graph = Graph("bolt://localhost:7687", auth=("TeaNLP", "teanlp"))

NEWLINE = '_NEWLINE_'


@app.route("/")
def hello():
    graph.run("Match () Return 1 Limit 1")
    return 'Hello, Amparo!'


@app.route("/tokenize", methods=['POST'])
def tokenize():
    tokens = []
    if request is not None and request.json is not None:
        print(request.json)
        request_json = json.loads(request.json, strict=False)
        if 'text' in request_json:
            text = request_json['text'].replace("\n", " " + NEWLINE + " ")
            tokens = TweetTokenizer().tokenize(text)
            for i in range(0, len(tokens)):
                if tokens[i] == NEWLINE:
                    tokens[i] = '\n'
    return {'result': tokens}

