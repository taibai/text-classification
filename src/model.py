import tensorflow as tf


def get_eval_metric_ops(predictions, labels):
    accuracy = tf.metrics.accuracy(predictions=predictions, labels=labels)

    return {
        "Accuracy": accuracy
    }


def get_model_fn(pretrained_word_embedding):
    def model_fn(features, labels, mode, params):

        def arch(inputs, num_hidden_units, num_attention_units, keep_prob, name):
            with tf.variable_scope(name):
                inputs = tf.nn.embedding_lookup(word_embedding, inputs)
                outputs = bilstm(inputs, seq_len, num_hidden_units,
                                 name='bilstm')
                outputs = attention(outputs, units=num_attention_units,
                                    name="attention")
                if mode == tf.estimator.ModeKeys.TRAIN:
                    outputs = tf.nn.dropout(outputs, keep_prob)
                return outputs

        head = features['head']
        tail = features['tail']
        seq_len = features['seq_len']

        word_embedding_ph = tf.placeholder(shape=pretrained_word_embedding.shape,
                                           dtype=pretrained_word_embedding.dtype)

        word_embedding = tf.get_variable('word_embedding', initializer=word_embedding_ph,
                                         trainable=False)

        head_outputs = arch(head, params.head_hidden_units, params.head_attention_units,
                            params.head_keep_prob)
        tail_outputs = arch(tail, params.tail_hidden_units, params.tail_attention_units,
                            params.tail_keep_prob)

        outputs = tf.concat((head_outputs, tail_outputs), -1)

        outputs = attention(outputs, params.attention_units, name='global_attention')

        with tf.name_scope('predict'):
            logits = tf.layers.dense(outputs, units=params.num_classes, use_bias=True)

            predictions = {
                'class': tf.argmax(tf.nn.softmax(logits), -1),
            }

        loss = None
        eval_metric_ops = {}
        train_op = None
        export_outputs = None

        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {'outputs': tf.estimator.export.PredictOutput(predictions)}

        if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):

            labels = tf.cast(labels, dtype=tf.int32)

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
            scaffold=tf.train.Scaffold(
                init_feed_dict={word_embedding_ph: pretrained_word_embedding})
        )

    return model_fn


def get_train_op(loss, learning_rate):
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        optimizer=tf.train.AdamOptimizer,
        learning_rate=learning_rate
    )


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


def get_serving_input_fn():
    features = {
        "words": tf.placeholder(dtype=tf.int32, shape=[None, None]),
        "word_seq_len": tf.placeholder(dtype=tf.int32, shape=[None]),
    }

    return tf.estimator.export.build_raw_serving_input_receiver_fn(features)
