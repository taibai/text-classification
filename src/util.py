import os

import numpy as np


def load_embedding(filename, dtype=np.float32):
    return np.load(filename).astype(dtype=dtype)


def readline(line):
    tokens = line.strip().split(',')
    char_sentences = [s.split() for s in tokens[0].split('#')]
    word_sentences = [s.split() for s in tokens[1].split('#')]
    label = int(tokens[-1]) - 1

    return char_sentences, word_sentences, label


def load_table(table_file):
    with open(table_file, 'r', encoding='UTF-8') as f:
        return {key: value for key, value in map(lambda x: x.strip().split(), f)}


def lookup(table, ids):
    if isinstance(ids, list):
        return [lookup(table, sub) for sub in ids]
    else:
        return table[ids] if ids in table else 0


def readfile(filename, max_doc_len, max_char_sen_len, max_word_sen_len):
    table_file = os.path.join(os.path.dirname(filename), 'table.txt')
    table = load_table(table_file)
    with open(filename, 'r', encoding='UTF-8') as f:
        for line in f:
            char_sentences, word_sentences, label = readline(line)

            doc_len = len(char_sentences) if len(char_sentences) <= max_doc_len else \
                max_doc_len

            word_sentences = lookup(table, word_sentences)

            char_sentences, char_sen_len = padding(char_sentences,
                                                   max_doc_len,
                                                   max_char_sen_len)
            word_sentences, word_sen_len = padding(word_sentences,
                                                   max_doc_len,
                                                   max_word_sen_len)
            yield char_sentences, word_sentences, char_sen_len, word_sen_len, doc_len, label


def padding(sentences, max_doc_len, max_sen_len):
    sen_len = [len(x) for x in sentences]

    if len(sen_len) < max_doc_len:
        sen_len += [0] * (max_doc_len - len(sen_len))

    sentences = [x + [0] * (max_sen_len - len(x)) for x in sentences]

    sentences = sentences + [[0] * max_sen_len] * (max_doc_len - len(sentences))

    return np.array(sentences, dtype=np.int32), np.array(sen_len, dtype=np.int32)


if __name__ == '__main__':
    filename = 'data/train.txt'

    count = 0
    try:
        for char_sentences, word_sentences, char_sen_len, word_sen_len, doc_len, label in readfile(
                filename, 128, 750, 175):
            count += 1
    except:
        print(count)

    print(count)
