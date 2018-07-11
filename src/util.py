import os

import numpy as np
import tensorflow as tf


def load_embedding(filename, dtype=np.float32):
    return np.load(filename).astype(dtype=dtype)


def readline(line):
    tokens = line.strip().split(',')
    # char_sentences = [s.split() for s in tokens[0].split('#')]
    word_sentences = [s.split() for s in tokens[1].split('#')]
    label = int(tokens[-1]) - 1

    # return char_sentences, word_sentences, label
    return word_sentences, label


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
            # char_sentences, word_sentences, label = readline(line)
            word_sentences, label = readline(line)

            word_sentences = word_sentences[:max_doc_len]

            # doc_len = len(word_sentences) if len(word_sentences) <= max_doc_len else \
            #     max_doc_len

            word_sentences = lookup(table, word_sentences)

            # char_sentences, char_sen_len = padding(char_sentences,
            #                                        max_doc_len,
            #                                        max_char_sen_len)
            word_sentences, word_sen_len = padding(word_sentences,
                                                   max_doc_len,
                                                   max_word_sen_len)
            # yield char_sentences, word_sentences, char_sen_len, word_sen_len, doc_len, label
            yield word_sentences, word_sen_len, label


def padding(sentences, max_doc_len, max_sen_len):
    sen_len = [len(x) for x in sentences]

    if len(sen_len) < max_doc_len:
        sen_len += [0] * (max_doc_len - len(sen_len))

    sentences = [x + [0] * (max_sen_len - len(x)) for x in sentences]

    sentences = sentences + [[0] * max_sen_len] * (max_doc_len - len(sentences))

    return np.array(sentences, dtype=np.int32), np.array(sen_len, dtype=np.int32)


def make_tfrecord_file(raw, max_doc_len, max_char_sen_len, max_word_sen_len, output):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    writer = tf.python_io.TFRecordWriter(output)
    for word_sentences, word_sen_len, label in readfile(raw, max_doc_len, max_char_sen_len,
                                                        max_word_sen_len):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'word_sentences': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[word_sentences.tostring()])),
                'word_sentences_shape': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=word_sentences.shape)),
                'word_sen_len': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[word_sen_len.tostring()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }
        ))

        serialized = example.SerializeToString()

        writer.write(serialized)

    writer.close()


def parse_fn(example_proto):
    dics = {
        'word_sentences': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'word_sen_len': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'word_sentences_shape': tf.FixedLenFeature(shape=(2, ), dtype=tf.int64),
        'label': tf.FixedLenFeature(shape=(), dtype=tf.int64),
    }

    parsed_example = tf.parse_single_example(example_proto, dics)

    parsed_example['word_sentences'] = tf.decode_raw(parsed_example['word_sentences'], tf.int32)
    parsed_example['word_sen_len'] = tf.decode_raw(parsed_example['word_sen_len'], tf.int32)

    parsed_example['word_sentences'] = tf.reshape(parsed_example['word_sentences'],
                                                  shape=parsed_example['word_sentences_shape'])
    return parsed_example['word_sentences'], parsed_example['word_sen_len'], parsed_example[
        'label']


if __name__ == '__main__':

    for filename in ('data/train.txt', 'data/eval.txt'):

        make_tfrecord_file(filename, 128, 750, 175, 'data/train.tfrecord')
