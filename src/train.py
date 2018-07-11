import functools
import os
import sys

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

from util import load_embedding
from model import get_model_fn, get_serving_input_fn
from inputs import input_fn
from config import Config

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'data_dir',
    'data',
    'data dir'
)

tf.app.flags.DEFINE_string(
    'model_dir',
    'model',
    'model dir'
)


def train(*args):
    params = Config()
    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig()

    run_config = run_config.replace(
        model_dir=FLAGS.model_dir,
        session_config=session_config
    )

    pretrained_char_embedding = load_embedding(os.path.join(FLAGS.data_dir, "char_embedding.npy"))
    pretrained_word_embedding = load_embedding(os.path.join(FLAGS.data_dir, "word_embedding.npy"))

    train_input_fn = functools.partial(input_fn,
                                       data_dir=FLAGS.data_dir,
                                       subset='train',
                                       batch_size=params.batch_size,
                                       num_epochs=None,
                                       shuffle=True)

    eval_input_fn = functools.partial(input_fn,
                                      data_dir=FLAGS.data_dir,
                                      subset='eval',
                                      batch_size=params.batch_size,
                                      num_epochs=None,
                                      shuffle=False)

    estimator = tf.estimator.Estimator(
        model_fn=get_model_fn(pretrained_char_embedding, pretrained_word_embedding),
        params=params,
        config=run_config
    )

    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=params.train_step
    )

    exporter = tf.estimator.FinalExporter(
        "classsification",
        get_serving_input_fn()
    )

    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        exporters=[exporter],
        name='classification_eval',
        steps=10,
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    tf.app.run(train, argv=[sys.argv[0]])
