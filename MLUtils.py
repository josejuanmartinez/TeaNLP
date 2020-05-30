import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer

from NLPUtils import NLPUtils
from config import NEWLINE


class MLUtils:

    @staticmethod
    def calculate_mean_embeddings(we):
        # we = PyTorch tensor of (1, N, 768), where N is num of tokens
        # we[0].shape[2] = 768 for BERT

        embeddings_dim = we.shape[2]
        tok_num = we.shape[1]

        tensor_acc = torch.tensor(np.zeros(embeddings_dim))

        for i in range(tok_num):
            tensor_acc += we[0][i]

        mean = tensor_acc / tok_num

        return mean

    @staticmethod
    def get_BERT_word_embeddings(text, only_lemmas=False):
        pretrained_weights = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        model = BertModel.from_pretrained(pretrained_weights)

        input = []

        for s, sentence in enumerate(NLPUtils.sentencize(text.strip())):
            toks, tokenizer = NLPUtils.tokenize(sentence, tokenizer)
            for t, tok in enumerate(toks):
                if tok == NEWLINE:
                    tok = '\n'

                if only_lemmas:
                    input.append(NLPUtils.lemmatize(tok))
                else:
                    input.append(tok)

        lemma_embeddings = torch.tensor(tokenizer.encode(' '.join(input), add_special_tokens=True)).unsqueeze(
            0)

        return model(lemma_embeddings)

    @staticmethod
    def get_BERT_sentence_embeddings(text):
        pretrained_weights = 'bert-base-nli-mean-tokens'
        sentence_model = SentenceTransformer(pretrained_weights)
        sentences = []
        for sentence in NLPUtils.sentencize(text):
            sentences.append(sentence)

        # This returns an array of tensors
        sentence_embeddings_numpy = sentence_model.encode(sentences)
        sentence_embeddings_tensors = []
        for sen in sentence_embeddings_numpy:
            sentence_embeddings_tensors.append(torch.tensor(sen))

        # I want a tensor instead
        ndim_tensor = torch.stack(sentence_embeddings_tensors)
        # print("Converting from array of numpy to stacked tensor")
        # print(ndim_tensor.shape)

        return ndim_tensor

    @staticmethod
    def get_embeddings(text, tok_num):

        # Word Embeddings
        # hidden_layer, we is a PyTorch tensor of (1, N, 768), where N is num of tokens
        # [0] because is last layer
        we = MLUtils.get_BERT_word_embeddings(text)[0]
        #print("Word embeddings tensors:")
        # print(we)
        # print(we.shape)

        if tok_num >= we.shape[1]:
            print ("Parameter 'tok_num' bigger than number of tokens. Returning None")
            we = None

        # Sentence Embeddings
        se = MLUtils.get_BERT_sentence_embeddings(text)
        # print("Sentence embeddings tensors:")
        # print(se)

        # Text embeddings: Mean of the word embeddings.
        te = MLUtils.calculate_mean_embeddings(we)
        # print("Text embeddings tensors:")
        # print(te)
        # print(te.shape)

        return we[0][tok_num], se, te
