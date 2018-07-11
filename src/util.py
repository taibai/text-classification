import os

import numpy as np
import tensorflow as tf


def load_embedding(filename, dtype=np.float32):
    return np.load(filename).astype(dtype=dtype)


def readline(line):
    tokens = line.strip().split(',')
    words = tokens[1].split()
    label = int(tokens[-1]) - 1

    # return char_sentences, word_sentences, label
    return words, label


def load_table(table_file):
    with open(table_file, 'r', encoding='UTF-8') as f:
        return {key: value for key, value in map(lambda x: x.strip().split(), f)}


def lookup(table, ids):
    if isinstance(ids, list):
        return [lookup(table, sub) for sub in ids]
    else:
        return table[ids] if ids in table else 0


def readfile(filename):
    table_file = os.path.join(os.path.dirname(filename), 'table.txt')
    table = load_table(table_file)
    with open(filename, 'r', encoding='UTF-8') as f:
        for line in f:
            # char_sentences, word_sentences, label = readline(line)
            words, label = readline(line)

            # doc_len = len(word_sentences) if len(word_sentences) <= max_doc_len else \
            #     max_doc_len

            words = lookup(table, words)

            # char_sentences, char_sen_len = padding(char_sentences,
            #                                        max_doc_len,
            #                                        max_char_sen_len)

            # yield char_sentences, word_sentences, char_sen_len, word_sen_len, doc_len, label
            yield np.array(words, dtype=np.int32), label


def make_tfrecord_file(raw, output):
    writer = tf.python_io.TFRecordWriter(output)
    for words, label in readfile(raw):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'words': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[words.tostring()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }
        ))

        serialized = example.SerializeToString()

        writer.write(serialized)

    writer.close()


def parse_fn(example_proto):
    dics = {
        'words': tf.FixedLenFeature(shape=(), dtype=tf.int64),
        'label': tf.FixedLenFeature(shape=(), dtype=tf.int64),
    }

    parsed_example = tf.parse_single_example(example_proto, dics)


    return parsed_example['words'], parsed_example['label']


if __name__ == '__main__':

    for subset in ('train', 'eval'):
        raw = f"data/{subset}.txt"
        output = f"data/{subset}.tfrecord"

        make_tfrecord_file(raw, output)
