import os
from multiprocessing.pool import Pool

import numpy as np


def load_embedding(filename, dtype=np.float32):
    return np.load(filename).astype(dtype=dtype)


def trans_embedding(filename):
    embedding = load_embedding(filename)

    zero = np.random.uniform(-0.8, 0.8, (1, embedding.shape[-1]))

    return np.concatenate((zero, embedding), axis=-1).astype(dtype=np.float32)


def readline(line):
    tokens = line.strip().split(',')
    char_sentences = [s.split() for s in tokens[0].split('#')]
    word_sentences = [s.split() for s in tokens[1].split('#')]
    label = int(tokens[-1])

    return char_sentences, word_sentences, label


def readfile(filename, **kwargs):
    with open(filename, 'r', encoding='UTF-8') as f:
        for line in f:
            char_sentences, word_sentences, label = readline(line)

            doc_len = len(char_sentences) if len(char_sentences) <= kwargs.max_doc_len else \
                kwargs.max_doc_len
            doc_len = doc_len + [0] * (kwargs.max_doc_len - len(doc_len))
            char_sentences, char_sen_len = padding(char_sentences,
                                                   kwargs.max_doc_len,
                                                   kwargs.max_char_sen_len)
            word_sentences, word_sen_len = padding(word_sentences,
                                                   kwargs.max_doc_len,
                                                   kwargs.max_word_sen_len)
            yield char_sentences, word_sentences, char_sen_len, word_sen_len, doc_len, label


def padding(sentences, max_doc_len, max_sen_len):
    sen_len = [len(x) for x in sentences]

    sentences = [[int(s) for s in x] + [0] * (max_sen_len - len(x)) for x in sentences]

    sentences = sentences + [[0] * max_sen_len] * (max_doc_len - len(sentences))

    return np.array(sentences, dtype=np.int32), np.array(sen_len, dtype=np.int32)
