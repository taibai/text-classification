import os
import numpy as np

import tensorflow as tf

from src.util import readfile


def get_file_name(data_dir, subset, suffix='txt'):
    if subset in ('train', 'eval', 'test'):
        return os.path.join(data_dir, f"{subset}.{suffix}")

    else:
        raise ValueError(f"Invalid data subset {subset}")


def input_fn(data_dir, subset, max_doc_len, max_sen_len, batch_size, num_epochs, shuffle=True):
    filename = get_file_name(data_dir, subset)

    dataset = get_dataset(filename, max_doc_len, max_sen_len, batch_size, num_epochs, shuffle)

    it = dataset.make_one_shot_iterator()

    char_sentences, word_sentences, char_sen_len, word_sen_len, doc_len, label = it.get_next()

    features = {
        "char_sentences": char_sentences,
        "char_sen_len": char_sen_len,
        "word_sentences": word_sentences,
        "word_sen_len": word_sen_len,
        "doc_len": doc_len,
    }

    labels = label

    return features, labels


def get_dataset(filename, max_char_sen_len, max_word_sen_len, max_doc_len, batch_size, num_epochs,
                shuffle):
    def generator():
        return readfile(filename, max_doc_len=max_doc_len, max_char_sen_len=max_char_sen_len,
                        max_word_sen_len=max_word_sen_len)

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_types=(tf.int32),
                                             output_shapes=([None, max_doc_len, max_char_sen_len],
                                                            [None, max_doc_len, max_word_sen_len],
                                                            [None, max_doc_len],
                                                            [None, max_doc_len],
                                                            [None], [None]))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.repeat(num_epochs)

    dataset = dataset.batch(batch_size)

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
