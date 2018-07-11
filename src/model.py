import tensorflow as tf

"""初步建议使用CNN + RNN 模型

    RNN 用于文档层面
"""


def get_eval_metric_ops(predictions, labels):
    accuracy = tf.metrics.accuracy(predictions=predictions, labels=labels)

    return {
        "Accuracy": accuracy
    }


def get_model_fn(pretrained_char_embedding, pretrained_word_embedding):
    def model_fn(features, labels, mode, params):
        # (batch_size, doc_len, seq_len, dim)
        # char_sentences = features['char_sentences']
        # char_sen_len = features['char_sen_len']
        word_sentences = features['word_sentences']
        word_sen_len = features['word_sen_len']
        doc_len = features['doc_len']

        batch_size = tf.shape(word_sen_len)[0]

        # char_sen_len = tf.reshape(char_sen_len, shape=[-1])
        # char_sentences = tf.reshape(char_sentences, shape=[-1, params.max_char_sen_len])

        # char_embedding_ph = tf.placeholder(shape=pretrained_char_embedding.shape,
        #                                    dtype=pretrained_char_embedding.dtype)

        # char_embedding = tf.get_variable('char_embedding', initializer=char_embedding_ph,
        #                                  trainable=False)

        # char_inputs = tf.nn.embedding_lookup(char_embedding, char_sentences)

        # char_bilstm_outputs = bilstm(char_inputs, char_sen_len, params.num_char_hidden_units,
        #                              name='char_bilstm')

        # char_att_outputs = attention(char_bilstm_outputs, units=params.char_attention_units,
        #                              name="char_sen_att")

        # char_dropout = tf.nn.dropout(char_att_outputs, keep_prob=params.dropout_char_keep_prob)

        word_embedding_ph = tf.placeholder(shape=pretrained_word_embedding.shape,
                                           dtype=pretrained_word_embedding.dtype)

        word_embedding = tf.get_variable('word_embedding', initializer=word_embedding_ph,
                                         trainable=False)

        word_sen_len = tf.reshape(word_sen_len, shape=[-1])
        word_sentences = tf.reshape(word_sentences, shape=[-1, params.max_word_sen_len])

        word_inputs = tf.nn.embedding_lookup(word_embedding, word_sentences)
        word_bilstm_outputs = bilstm(word_inputs, word_sen_len, params.num_word_hidden_units,
                                     name='word_bilstm')

        word_att_outputs = attention(word_bilstm_outputs, units=params.word_attention_units,
                                     name="word_sen_att")

        word_dropout = tf.nn.dropout(word_att_outputs, keep_prob=params.dropout_word_keep_prob)

        # outputs = merge(char_dropout, word_dropout, params)
        outputs = tf.reshape(word_dropout, shape=[batch_size, -1, 2 * params.num_word_hidden_units])

        outputs = bilstm(outputs, doc_len, params.num_doc_hidden_units, name='document_bilstm')

        outputs = attention(outputs, params.doc_attention_units, name='document_attention')

        outputs = tf.nn.dropout(outputs, keep_prob=params.dropout_doc_keep_prob)

        with tf.name_scope('predict'):
            logits = tf.layers.dense(outputs, units=params.num_classes, use_bias=True)

            predictions = {
                'class': tf.argmax(tf.nn.softmax(logits), 1) + 1,
            }

        loss = None
        eval_metric_ops = {}
        train_op = None
        export_outputs = None

        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {'outputs': tf.estimator.export.PredictOutput(predictions)}

        if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):

            y = tf.one_hot(labels, params.num_classes)

            with tf.variable_scope("losses"):
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
                loss = tf.reduce_mean(loss)
                loss = tf.identity(loss, name='loss')
                if mode == tf.estimator.ModeKeys.TRAIN:
                    with tf.name_scope("optimizer"):
                        train_op = get_train_op(loss, params.learning_rate)
                if mode == tf.estimator.ModeKeys.EVAL:
                    eval_metric_ops = get_eval_metric_ops(predictions['class'], labels)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            export_outputs=export_outputs,
            eval_metric_ops=eval_metric_ops,
            scaffold=tf.train.Scaffold(init_feed_dict={word_embedding_ph:
                                                           pretrained_word_embedding})
                                                       # char_embedding_ph: pretrained_char_embedding})
        )

    return model_fn


def get_train_op(loss, learning_rate):
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        optimizer=tf.train.AdamOptimizer,
        learning_rate=learning_rate
    )


def merge(char_att_outputs, word_att_outputs, params):
    char_inputs = tf.reshape(char_att_outputs,
                             shape=[-1, params.max_doc_len,
                                    2 * params.num_char_hidden_units])
    word_inputs = tf.reshape(word_att_outputs,
                             shape=[-1, params.max_doc_len,
                                    2 * params.num_word_hidden_units])
    outputs = tf.concat((char_inputs, word_inputs), axis=-1)
    return outputs


def attention(inputs, units, name):
    with tf.variable_scope(name):
        g = tf.layers.dense(inputs, units=units, use_bias=True)

        # omega = tf.get_variable('omega', shatf.random_normal([units], stddev=0.1))
        omega = tf.get_variable('omega', shape=[units], initializer=tf.random_normal_initializer(
            stddev=0.1))

        alphas = tf.nn.softmax(tf.tensordot(g, omega, axes=1), name='alphas')

        outputs = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        return outputs


def bilstm(inputs, sequence_length, num_hidden_units, name):
    with tf.variable_scope(name):
        with tf.variable_scope('forwords'):
            cell_fw = tf.contrib.rnn.BasicLSTMCell(num_hidden_units)
        with tf.variable_scope('backwords'):
            cell_bw = tf.contrib.rnn.BasicLSTMCell(num_hidden_units)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                     sequence_length=sequence_length,
                                                     dtype=tf.float32)
        return tf.concat(outputs, -1)


def get_serving_input_fn(max_doc_len, max_char_sen_len, max_word_sen_len):
    features = {
        # "char_sentences": tf.placeholder(dtype=tf.int32, shape=[None, max_doc_len,
        #                                                         max_char_sen_len]),
        # "char_sen_len": tf.placeholder(dtype=tf.int32, shape=[None, max_doc_len]),
        "word_sentences": tf.placeholder(dtype=tf.int32, shape=[None, max_doc_len,
                                                                max_word_sen_len]),
        "word_sen_len": tf.placeholder(dtype=tf.int32, shape=[None, max_doc_len]),
        "doc_len": tf.placeholder(dtype=tf.int32, shape=[None]),
    }

    return tf.estimator.export.build_raw_serving_input_receiver_fn(features)
