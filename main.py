import base64
import copy
import io
import json

import torch
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

import json

from domain.bert_subwords import BertSubwords
from domain.offset import Offset
from domain.token import Token

app = Flask(__name__)
CORS(app)
graph = Graph("bolt://localhost:7687", auth=("TeaNLP", "teanlp"))

nlputils = NLPUtils()
mlutils = MLUtils()


@app.route("/")
def hello():
    graph.run("Match () Return 1 Limit 1")
    return 'Hello, Amparo!'


@app.route("/preprocess", methods=['POST'])
def preprocess():
    preprocessed_tokens = []
    if request is not None and request.json is not None:
        request_json = json.loads(request.json, strict=False)
        if 'text' in request_json:

            text = request_json['text'].replace("\n", " " + NEWLINE + " ")
            # print("Text: {}".format(text))
            text_embeddings = MLUtils.get_bert_text_embeddings(text)
            # print("Text Embeddings: {}".format(text_embeddings.shape))

            bio = io.BytesIO()
            torch.save(text_embeddings, bio)
            b64_text_embeddings = str(base64.b64encode(bio.getvalue()))
            # print(b64_text_embeddings)
            bio.close()

            tok_counter = 0
            text_tok_counter_original = 0
            text_tok_counter_lower = 0

            for s, sentence in enumerate(NLPUtils.sentencize(text)):
                # print("Sentence: {}".format(sentence))
                original_merged_tokens, lower_merged_tokens, truecased_merged_tokens = MLUtils.subword_tokenize(sentence)
                # print("Original tokens[{}]: {}".format(len(original_merged_tokens), original_merged_tokens))
                # print("Lower tokens[{}]: {}".format(len(lower_merged_tokens), lower_merged_tokens))
                original_clean_tokens = MLUtils.subwords_to_words(truecased_merged_tokens)
                # print("Clean tokens[{}]: {}".format(len(original_clean_tokens), original_clean_tokens))

                sentence_embeddings = MLUtils.get_bert_sentence_embeddings(sentence)

                bio = io.BytesIO()
                torch.save(sentence_embeddings, bio)
                b64_sentence_embeddings = str(base64.b64encode(bio.getvalue()))
                # print(b64_sentence_embeddings)
                bio.close()

                original_word_embeddings = MLUtils.get_bert_word_embeddings(sentence)[0]

                lower_word_embeddings = MLUtils.get_bert_word_embeddings(sentence.lower())[0]

                pos = nltk.pos_tag(original_clean_tokens)
                sent_tok_counter = 0
                sent_tok_counter_original = 0
                sent_tok_counter_lower = 0
                for t, token in enumerate(original_clean_tokens):
                    if token == NEWLINE:
                        token = '\n'

                    bio = io.BytesIO()
                    torch.save(original_word_embeddings[sent_tok_counter_original], bio)
                    b64_original_word_embeddings = str(base64.b64encode(bio.getvalue()))
                    bio.close()

                    bio = io.BytesIO()
                    torch.save(lower_word_embeddings[sent_tok_counter_lower], bio)
                    b64_lower_word_embeddings = str(base64.b64encode(bio.getvalue()))
                    bio.close()

                    offset = Offset(s, text_tok_counter_original, tok_counter, text_tok_counter_lower, t)
                    original_bert_subwords = BertSubwords(original_merged_tokens[sent_tok_counter],
                                                          b64_original_word_embeddings)
                    lower_bert_subwords = BertSubwords(lower_merged_tokens[sent_tok_counter],
                                                       b64_lower_word_embeddings)

                    tok = Token(token, pos[t][1], offset, original_bert_subwords, lower_bert_subwords,
                                b64_sentence_embeddings, b64_text_embeddings, original_clean_tokens)

                    preprocessed_tokens.append(tok.to_json())

                    text_tok_counter_original += original_bert_subwords.length
                    text_tok_counter_lower += lower_bert_subwords.length
                    sent_tok_counter_original += original_bert_subwords.length
                    sent_tok_counter_lower += lower_bert_subwords.length

                    sent_tok_counter += 1
                    tok_counter += 1
    else:
        return 'Bad request.', 400

    res_json = json.dumps({'result': preprocessed_tokens}, indent=4)
    print(res_json)

    return res_json

"""
@app.route("/save", methods=['POST'])
def save():
    if request is not None and request.json is not None:
        request_json = json.loads(request.json, strict=False)
        if 'text' in request_json:
            text = request_json['text'].replace("\n", " " + NEWLINE + " ")
        else:
            return "'text' not present in POST request", 400
        if 'token' in request_json:
            token = request_json['token']

            tok_order = int(token[BERT_SUBWORDS_ORIGINAL_START])
            tok_sentence = int(token[SENTENCE])
        else:
            return "'tok_feat' not present in POST request", 400
    else:
        return 'Bad request.', 400

    bio = io.BytesIO()
    torch.save(we, bio)
    print(base64.b64encode(bio.getvalue()))
    bio.close()

    bio = io.BytesIO()
    torch.save(se, bio)
    print(base64.b64encode(bio.getvalue()))
    bio.close()

    bio = io.BytesIO()
    torch.save(te, bio)
    print(base64.b64encode(bio.getvalue()))
    bio.close()

    return {'acknowledged': True}
"""
