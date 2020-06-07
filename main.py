import base64
import io

import torch
from flask import Flask
from flask import request
from flask_cors import CORS

import nltk
from py2neo import Node, Relationship

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

                    # Symbolic NER
                    if NLPUtils.is_email(tok.linguistic_features.orth):
                        tok.statistical_features.bert_subwords_original.ner.add('I-EMAIL')
                    if tok.linguistic_features.is_num:
                        tok.statistical_features.bert_subwords_original.ner.add('I-NUM')
                    if NLPUtils.is_currency(tok.linguistic_features.orth):
                        tok.statistical_features.bert_subwords_original.ner.add('I-CURRENCY')

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

            tx = Neo4JUtils.graph().begin()



            tok = Neo4JUtils.find_one("Token", property_key="orth",
                                                       property_value=token['linguistic_features']['orth'])

            if not tok:
                tok = Node("Token", orth=token['linguistic_features']['orth'],
                             is_stop=token['linguistic_features']['is_stop'],
                             is_punct=token['linguistic_features']['is_punct'],
                             is_space=token['linguistic_features']['is_space'],
                             meaningful_embedding=token['linguistic_features']['meaningful_embedding'],
                             is_num=token['linguistic_features']['is_num'],
                             is_alpha=token['linguistic_features']['is_alpha'])

                tx.create(tok)

            embedding = Neo4JUtils.find_one("WordEmbedding", property_key="embedding",
                                                property_value=token['statistical_features']['bert_subwords_original']
                                                ['embedding'])

            if not embedding:
                embedding = Node("WordEmbedding", subwords=token['statistical_features']['bert_subwords_original']['subwords'],
                             subwords_root=token['statistical_features']['bert_subwords_original']['root'],
                             subwords_length=token['statistical_features']['bert_subwords_original']['length'],
                             is_subwords=token['statistical_features']['bert_subwords_original']['is_subwords'],
                             embedding=token['statistical_features']['bert_subwords_original']['embedding'])
                tx.create(embedding)


            lower = Neo4JUtils.find_one("Token", property_key="orth",
                                                       property_value=token['linguistic_features']['lower'])

            if not lower:
                lower = Node("Token", orth=token['linguistic_features']['lower'],
                         is_stop=token['linguistic_features']['is_stop'],
                         is_punct=token['linguistic_features']['is_punct'],
                         is_space=token['linguistic_features']['is_space'],
                         meaningful_embedding=token['linguistic_features']['meaningful_embedding'],
                         is_num=token['linguistic_features']['is_num'],
                         is_alpha=token['linguistic_features']['is_alpha'])
                tx.create(lower)


            lower_embedding = Neo4JUtils.find_one("WordEmbedding", property_key="embedding",
                                                property_value=token['statistical_features']['bert_subwords_lower']
                                                ['embedding'])

            if not lower_embedding:
                lower_embedding = Node("WordEmbedding",
                             subwords=token['statistical_features']['bert_subwords_lower']['subwords'],
                             subwords_root=token['statistical_features']['bert_subwords_lower']['root'],
                             subwords_length=token['statistical_features']['bert_subwords_lower']['length'],
                             is_subwords=token['statistical_features']['bert_subwords_lower']['is_subwords'],
                             embedding=token['statistical_features']['bert_subwords_lower']['embedding'])
                tx.create(lower_embedding)

            stem = Neo4JUtils.find_one("Stem", property_key="value",
                                               property_value=token['linguistic_features']['stem'])

            if not stem:
                stem = Node("Stem", value=token['linguistic_features']['stem'])
                tx.create(stem)

            lemma = Neo4JUtils.find_one("Token", property_key="orth",
                                               property_value=token['linguistic_features']['lemma'])

            if not lemma:
                lemma = Node("Token", orth=token['linguistic_features']['lemma'])
                tx.create(lemma)

            se = Neo4JUtils.find_one("SentenceEmbedding", property_key="value",
                                             property_value=token['statistical_features']['sentence_embedding'])

            if not se:
                se = Node("SentenceEmbedding", value=token['statistical_features']['sentence_embedding'])
                tx.create(se)

            te = Neo4JUtils.find_one("TextEmbedding", property_key="value",
                                             property_value=token['statistical_features']['text_embedding'])

            if not te:
                te = Node("TextEmbedding", value=token['statistical_features']['text_embedding'])
                tx.create(te)

            pos = Neo4JUtils.find_one("PoS", property_key="value",
                                              property_value=token['linguistic_features']['pos'])

            if not pos:
                pos = Node("PoS", value=token['linguistic_features']['pos'])
                tx.create(pos)

            r = Relationship(tok, "HAS_LOWER", lower)

            tx.merge(r)

            r = Relationship(tok, "HAS_STEM", stem)

            tx.merge(r)

            r = Relationship(tok, "HAS_LEMMA", lemma)

            tx.merge(r)

            r = Relationship(tok, "HAS_SENTENCE_EMBEDDINGS", se)

            tx.merge(r)

            r = Relationship(tok, "HAS_TEXT_EMBEDDINGS", te)

            tx.merge(r)

            r = Relationship(tok, "HAS_POS", pos)

            tx.merge(r)

            r = Relationship(tok, "HAS_WORD_EMBEDDING", embedding)

            tx.merge(r)

            r = Relationship(tok, "HAS_WORD_EMBEDDING", lower_embedding)

            tx.merge(r)

            if token['statistical_features']['bert_subwords_original']['ner'] is not None:
                for ner in token['statistical_features']['bert_subwords_original']['ner']:
                    if MLUtils.NOENT in ner:
                        continue

                    ner = Neo4JUtils.find_one("NER", property_key="value",
                                              property_value=ner)

                    if not ner:
                        ner = Node("NER", value=ner)
                        tx.create(ner)


            if token['statistical_features']['bert_subwords_lower']['ner'] is not None:
                for ner in token['statistical_features']['bert_subwords_lower']['ner']:
                    if MLUtils.NOENT in ner:
                        continue

                    ner = Neo4JUtils.find_one("NER", property_key="value",
                                              property_value=ner)

                    if not ner:
                        ner = Node("NER", value=ner)
                        tx.create(ner)

            tx.commit()

            return json.dumps({'result': 'acknowledge'})
    return json.dumps({'result': 'error'})

