import base64
import io

import torch
from flask import Flask
from flask import request
from flask_cors import CORS

import nltk

from MLUtils import MLUtils
from NLPUtils import NLPUtils
from Neo4JUtils import Neo4JUtils
from config import *

import json

from domain.bert_subwords import BertSubwords
from domain.offset import Offset
from domain.token import Token

app = Flask(__name__)
CORS(app)

nlputils = NLPUtils()
mlutils = MLUtils()
neo4jutils = Neo4JUtils()

@app.route("/neo4j-health")
def hello():
    return Neo4JUtils.health()


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

                # 1) I tokenize sentence and sentence.lower()
                original_tokens = MLUtils.tokenize(sentence)
                # 2) I do statistical NER (BERT)
                ner = MLUtils.ner(sentence, grouped_entities=False)
                original_merged_tokens = MLUtils.merge_subwords(original_tokens, ner)

                # 3) I merge subwords (## for BERT)
                lower_tokens = MLUtils.tokenize(sentence.lower())
                lower_merged_tokens = MLUtils.merge_subwords(lower_tokens)

                # 3) I truecase the lower subwords so that they keep original capitalization
                truecased_merged_tokens = NLPUtils.truecase(original_merged_tokens, lower_merged_tokens)

                # 4) I clean the subwords mark (## for BERT).
                original_words = [x['text'] for x in truecased_merged_tokens]
                original_clean_words = MLUtils.subwords_to_words(original_words)
                # original_clean_nums = [x['num'] for x in truecased_merged_tokens]
                original_clean_ners = [x['ner'] for x in truecased_merged_tokens]


                # Sentence Embeddings
                sentence_embeddings = MLUtils.get_bert_sentence_embeddings(sentence)

                bio = io.BytesIO()
                torch.save(sentence_embeddings, bio)
                b64_sentence_embeddings = str(base64.b64encode(bio.getvalue()))
                bio.close()

                original_word_embeddings = MLUtils.get_bert_word_embeddings(sentence)[0]

                lower_word_embeddings = MLUtils.get_bert_word_embeddings(sentence.lower())[0]

                pos = nltk.pos_tag(original_clean_words)
                sent_tok_counter = 0
                sent_tok_counter_original = 0
                sent_tok_counter_lower = 0
                for t, token in enumerate(original_clean_words):
                    if token == NEWLINE:
                        token = '\n'

                    b64_original_word_embeddings = ''
                    b64_lower_word_embeddings = ''

                    if not NLPUtils.is_space(token) and token.strip() != '' and MLUtils.SUBWORD_MARK not in token:

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
                                                          b64_original_word_embeddings, original_clean_ners[t])
                    # Lower subwords don't have reliable NER entities
                    lower_bert_subwords = BertSubwords(lower_merged_tokens[sent_tok_counter],
                                                       b64_lower_word_embeddings, ner=None)

                    tok = Token(token, pos[t][1], offset, original_bert_subwords, lower_bert_subwords,
                                b64_sentence_embeddings, b64_text_embeddings, original_clean_words)

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


@app.route("/save", methods=['POST'])
def save():
    if request is not None and request.json is not None:
        request_json = json.loads(request.json, strict=False)
        if 'token' in request_json:
            token = request_json['token']

            print(token)

            # NEO4J Query Nodes
            query = "MERGE (token:Token {{orth: '{}', is_stop: {}, is_punct: {}, is_space: {}, " \
                    "meaningful_embedding: {}, is_num: {}, is_alpha: {}}})\n".format(
                token['linguistic_features']['orth'].replace("'", "\\'"),
                token['linguistic_features']['is_stop'],
                token['linguistic_features']['is_punct'],
                token['linguistic_features']['is_space'],
                token['linguistic_features']['meaningful_embedding'],
                token['linguistic_features']['is_num'],
                token['linguistic_features']['is_alpha'])

            query += "MERGE (embedding:WordEmbedding {{subwords: '{}', subwords_root: '{}', " \
                    "subwords_length: {}, is_subwords: {}, embedding: '{}'}})\n".format(
                token['statistical_features']['bert_subwords_original']['subwords'].replace("'", "\\'"),
                token['statistical_features']['bert_subwords_original']['root'].replace("'", "\\'"),
                token['statistical_features']['bert_subwords_original']['length'],
                token['statistical_features']['bert_subwords_original']['is_subwords'],
                token['statistical_features']['bert_subwords_original']['embedding'].replace("'", "\\'"))

            query += "MERGE (lower:Token {{orth: '{}', is_stop: {}, is_punct: {}, is_space: {}, " \
                    "meaningful_embedding: {}, is_num: {}, is_alpha: {}}})\n".format(
                token['linguistic_features']['lower'].replace("'", "\\'"),
                token['linguistic_features']['is_stop'],
                token['linguistic_features']['is_punct'],
                token['linguistic_features']['is_space'],
                token['linguistic_features']['meaningful_embedding'],
                token['linguistic_features']['is_num'],
                token['linguistic_features']['is_alpha'])

            query += "MERGE (lowerEmbedding:WordEmbedding {{subwords: '{}', subwords_root: '{}', " \
                    "subwords_length: {}, is_subwords: {}, embedding: '{}'}})\n".format(
                token['statistical_features']['bert_subwords_lower']['subwords'].replace("'", "\\'"),
                token['statistical_features']['bert_subwords_lower']['root'].replace("'", "\\'"),
                token['statistical_features']['bert_subwords_lower']['length'],
                token['statistical_features']['bert_subwords_lower']['is_subwords'],
                token['statistical_features']['bert_subwords_lower']['embedding'].replace("'", "\\'"))

            query += "MERGE (stem:Token {{orth: '{}'}})\n".format(token['linguistic_features']['stem'].replace("'", "\\'"))

            query += "MERGE (lemma:Token {{orth: '{}'}})\n".format(token['linguistic_features']['lemma'].replace("'", "\\'"))

            query += "MERGE (se:SentenceEmbedding {{value: '{}'}})\n".format(token['statistical_features']['sentence_embedding'].replace("'", "\\'"))

            query += "MERGE (te:TextEmbedding {{value: '{}'}})\n".format(token['statistical_features']['text_embedding'].replace("'", "\\'"))

            query += "MERGE (pos:POS {{value: '{}'}})\n".format(token['linguistic_features']['pos'])

            Neo4JUtils.run(query)

            query = "MATCH (token: Token), " \
                    "(stem: Token), " \
                    "(lemma: Token), "\
                    "(lower: Token), "\
                    "(se: SentenceEmbedding), "\
                    "(te: TextEmbedding), "\
                    "(pos: POS), "\
                    "(wordEmbedding: WordEmbedding), "\
                    "(lowerEmbedding: WordEmbedding)\n"\
                    "WHERE token.orth='{}' AND " \
                    "stem.orth='{}' AND " \
                    "lower.orth='{}' AND "\
                    "lemma.orth='{}' AND " \
                    "se.value='{}' AND " \
                    "te.value='{}' AND " \
                    "pos.value='{}' AND " \
                    "wordEmbedding.embedding='{}' AND " \
                    "lowerEmbedding.embedding='{}'\n" \
                    "MERGE (token)-[:IS_LOWER]->(lower)\n" \
                    "MERGE (token)-[:IS_STEM]->(stem)\n" \
                    "MERGE (token)-[:IS_LEMMA]->(lemma)\n" \
                    "MERGE (token)-[:HAS_SENTENCE_EMBEDDINGS]->(se)\n" \
                    "MERGE (token)-[:HAS_TEXT_EMBEDDINGS]->(te)\n" \
                    "MERGE (token)-[:HAS_POS]->(pos)\n" \
                    "MERGE (token)-[:HAS_WORD_EMBEDDING]->(wordEmbedding)\n" \
                    "MERGE (token)-[:HAS_WORD_EMBEDDING]->(lowerEmbedding)".format(token['linguistic_features']['orth'].replace("'", "\\'"),
                                                                            token['linguistic_features']['stem'].replace("'", "\\'"),
                                                                            token['linguistic_features']['lower'].replace("'", "\\'"),
                                                                            token['linguistic_features']['lemma'].replace("'", "\\'"),
                                                                            token['statistical_features']
                                                                            ['sentence_embedding'].replace("'", "\\'"),
                                                                            token['statistical_features']
                                                                            ['text_embedding'].replace("'", "\\'"),
                                                                            token['linguistic_features']['pos'],
                                                                            token['statistical_features']
                                                                            ['bert_subwords_original']['embedding'].replace("'", "\\'"),
                                                                            token['statistical_features']
                                                                            ['bert_subwords_lower']['embedding'].replace("'", "\\'"))

            Neo4JUtils.run(query)

            if token['statistical_features']['bert_subwords_original']['ner'] is not None:
                for ner in token['statistical_features']['bert_subwords_original']['ner']:
                    if MLUtils.NOENT in ner:
                        continue

                    query = "MATCH (token:Token) WHERE token.orth='{}'\n" \
                            "MERGE (ner:NER {{value: '{}'}})\n" \
                            "MERGE (token)-[:HAS_NER]->(ner)".format(token['linguistic_features']['orth'],
                                                              ner.replace("'", "\\'"))

                    Neo4JUtils.run(query)

            if token['statistical_features']['bert_subwords_lower']['ner'] is not None:
                for ner in token['statistical_features']['bert_subwords_lower']['ner']:
                    if MLUtils.NOENT in ner:
                        continue

                    query = "MATCH (token:Token) WHERE token.orth='{}'\n" \
                            "MERGE (lowerner:NER {{value: '{}'}})\n" \
                            "MERGE (token)-[:HAS_NER]->(ner)".format(token['linguistic_features']['orth'],
                                                              ner.replace("'", "\\'"))

                    Neo4JUtils.run(query)

            return json.dumps({'result': 'acknowledge'})
    return json.dumps({'result': 'error'})

