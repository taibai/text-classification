import tensorflow as tf

"""初步建议使用CNN + RNN 模型

    RNN 用于文档层面
"""


def model_fn(features, labels, mode, params):
    # (batch_size, doc_len, seq_len, dim)
    char_sentences = features['char_sentences']
    char_sen_len = features['char_sen_len']
    word_sentences = features['word_sentences']
    word_sen_len = features['word_sen_len']
    doc_len = features['doc_len']

    char_sen_len = tf.reshape(char_sen_len, shape=[-1])
    char_sentences = tf.reshape(char_sentences, shape=[-1, params.max_char_sen_len])

    char_inputs = embedding(char_sentences, params, name='char_embedding')

    char_bilstm_outputs = bilstm(char_inputs, char_sen_len, params, name='char_bilstm')

    char_att_outputs = attention(char_bilstm_outputs, units=params.char_attention_units,
                                 name="char_sen_att")

    word_inputs = embedding(word_sentences, params, name='word_embedding')
    word_bilstm_outputs = bilstm(word_inputs, word_sen_len, params, name='word_bilstm')

    word_att_outputs = attention(word_bilstm_outputs, units=params.word_attention_units,
                                 name="word_sen_att")

    outputs = merge(char_att_outputs, word_att_outputs, params)

    outputs = bilstm(outputs, doc_len, params, name='document_bilstm')

    outputs = attention(outputs, params.doc_attention_units, params, name='document_attention')

    with tf.name_scope('predict'):
        logits = tf.layers.dense(outputs, units=params.num_classes, use_bias=True,
                                 activation=tf.nn.softmax)

    with tf.name_scope("optimize"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(loss)


def merge(char_att_outputs, word_att_outputs, params):
    char_inputs = tf.reshape(char_att_outputs,
                             shape=[-1, params.max_doc_len,
                                    params.max_char_sen_len * params.char_attention_units])
    word_inputs = tf.reshape(word_att_outputs,
                             shape=[-1, params.max_doc_len,
                                    params.max_word_sen_len * params.word_attention_units])
    outputs = tf.concat((char_inputs, word_inputs), axis=-1)
    return outputs


def attention(inputs, units, params, name):
    with tf.name_scope(name):
        g = tf.layers.dense(inputs, units=units)

        outputs = tf.layers.dense(tf.concat([inputs, g], -1), units=params.num_attention_units,
                                  activation=tf.nn.tanh)
    return outputs


def embedding(char_sentences, params, name):
    with tf.name_scope(name):
        embedding = tf.get_variable('embedding', [params.word_vocab_size,
                                                  params.word_embedding_dim])
        return tf.nn.embedding_lookup(embedding, char_sentences)


def bilstm(inputs, sequence_length, params, name):
    with tf.name_scope(name):
        with tf.variable_scope('forwords'):
            cell_fw = tf.nn.rnn_cell_impl.BasicLSTMCell(params.num_rnn_units)
        with tf.variable_scope('backwords'):
            cell_bw = tf.nn.rnn_cell_impl.BasicLSTMCell(params.num_rnn_units)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                     sequence_length=sequence_length,
                                                     dtype=tf.float32)
        return tf.concat(outputs, -1)
