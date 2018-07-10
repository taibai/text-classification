class Config:

    def __init__(self):
        self.max_char_sen_len = 750
        self.max_word_sen_len = 175
        self.max_doc_len = 128

        self.num_char_hidden_units = 64
        self.num_word_hidden_units = 64
        self.num_doc_hidden_units = 64

        self.char_attention_units = 5
        self.word_attention_units = 10
        self.doc_attention_units = 10

        self.dropout_char_keep_prob = 0.8
        self.dropout_word_keep_prob = 0.8
        self.dropout_doc_keep_prob = 0.8

        self.learning_rate = 0.001
        self.batch_size = 8

        self.train_step = 440000

        self.num_classes = 19
