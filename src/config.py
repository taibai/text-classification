class Config:

    def __init__(self):

        self.head_hidden_units = 64
        self.tail_hidden_units = 64

        self.head_attention_units = 5
        self.tail_attention_units = 10
        self.attention_units = 10

        self.head_keep_prob = 0.8
        self.tail_keep_prob = 0.8

        self.learning_rate = 0.001
        self.batch_size = 8

        self.train_step = 440000

        self.num_classes = 19
