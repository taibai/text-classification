import os

import numpy as np
import tensorflow as tf


def load_embedding(filename, dtype=np.float32):
    return np.load(filename).astype(dtype=dtype)


def readline(line):
    tokens = line.strip().split(',')
    head = tokens[0].split()
    tail = tokens[1].split()
    label = int(tokens[-1]) - 1

    # return char_sentences, word_sentences, label
    return head, tail, label


def load_table(table_file):
    with open(table_file, 'r', encoding='UTF-8') as f:
        return {key: int(value) for key, value in map(lambda x: x.strip().split(), f)}


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
            head, tail, label = readline(line)
            yield lookup(table, head), lookup(table, tail), label


def make_tfrecord_file(raw, output):
    writer = tf.python_io.TFRecordWriter(output)
    for head, tail, label in readfile(raw):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'head': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=head)),
                'tail': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=tail)),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }
        ))

        serialized = example.SerializeToString()

        writer.write(serialized)

    writer.close()


def parse_fn(example_proto):
    dics = {
        'head': tf.VarLenFeature(dtype=tf.int64),
        'tail': tf.VarLenFeature(dtype=tf.int64),
        'label': tf.FixedLenFeature(shape=(), dtype=tf.int64),
    }

    parsed_example = tf.parse_single_example(example_proto, dics)

    head = tf.sparse_tensor_to_dense(parsed_example['head'])
    tail = tf.sparse_tensor_to_dense(parsed_example['tail'])

    return head, tail, parsed_example['label']


if __name__ == '__main__':

    for subset in ('train', 'eval'):
        raw = f"data/{subset}.txt"
        output = f"data/{subset}.tfrecord"

        make_tfrecord_file(raw, output)
