import os
import numpy as np

import tensorflow as tf

from util import readfile, parse_fn


def bucketing(dataset, mini_batch_size):
    head_batch, tail_batch, label_batch = dataset.make_one_shot_iterator().get_next()

    with tf.variable_scope("bucketing"):
        _batch_size = tf.get_variable(name='mini_batch_size',
                                      dtype=tf.int32,
                                      initializer=0,
                                      trainable=False)
        _batch_size = tf.assign(ref=_batch_size, value=mini_batch_size)

        seq_len, [head, tail, labels] = tf.contrib.training.bucket_by_sequence_length(
            input_length=tf.shape(head_batch)[0],
            tensors=[head_batch, tail_batch, label_batch],
            batch_size=_batch_size,
            bucket_boundaries=[10, 30, 50, 80, 110],
            dynamic_pad=True,
        )

        features = {
            "head": head,
            "tail": tail,
            "seq_len": seq_len,
        }

        return features, labels


def get_file_name(data_dir, subset, suffix='txt'):
    if subset in ('train', 'eval', 'test'):
        return os.path.join(data_dir, f"{subset}.{suffix}")

    else:
        raise ValueError(f"Invalid data subset {subset}")


def input_fn(data_dir, subset, batch_size, num_epochs, shuffle=True):
    filename = get_file_name(data_dir, subset, suffix='tfrecord')

    dataset = get_dataset_from_tfrecord_file(filename, num_epochs, shuffle)

    return bucketing(dataset, batch_size)


def get_dataset(filename, num_epochs, shuffle):
    dataset = tf.data.Dataset.from_generator(lambda: readfile(filename),
                                             output_types=(tf.int32, tf.int32),
                                             output_shapes=([None], []))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.repeat(num_epochs)

    dataset = dataset.prefetch(1)

    return dataset


def get_dataset_from_tfrecord_file(filename, num_epochs, shuffle):
    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.map(parse_fn)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(1)

    return dataset


def process(line, max_sen_len, max_doc_len, s=','):
    label, *sentences = line.strip().split(s)

    sentences = [x.split("\t") for x in sentences]

    doc_len = len(sentences)

    sen_len = [len(x) for x in sentences] + [0] * (max_doc_len - len(doc_len))

    sentences = [[int(word) for word in x] + [0] * (max_sen_len - len(x)) for x in sentences]

    document = np.array(sentences + [[0] * max_sen_len] * (max_doc_len - len(sentences)),
                        dtype=np.int32)

    return document, label, doc_len, sen_len
