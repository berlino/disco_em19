from __future__ import division


class Config():
    def __init__(self):
        self.root_path = "."
        self.if_shuffle = True
        self.seed = 666

        # for NN only
        self.if_margin = False
        self.beta = 3

        # override when loading data
        self.voc_size = None
        self.char_voc_size = None
        self.label_size = 1
        self.class_weight = 1

        # embed size
        self.token_embed = 200
        self.char_embed = 32
        self.input_dropout = 0.6

        self.c_hidden_size = 32
        self.c_layers = 1
        self.c_lstm_dropout = 0 # [0,0.5]

        # for word_lstm
        self.f_hidden_size = 128  # 32, 64, 128, 256
        self.f_layers = 1
        self.f_lstm_dropout = 0.0 # [0,0.5]
        self.semi_hidden_size = self.f_hidden_size

        # disco classifier
        self.d_hidden_size = 6
        self.d_dropout = 0.0

        # for training
        self.embed_path = self.root_path + "/data/word_vec_{0}.pkl".format(self.token_embed)
        self.epoch = 500
        self.if_gpu = False
        self.opt = "Adam"
        self.lr = 0.005 # [0.3, 0.00006]
        self.l2 = 1e-4
        self.check_every = 1
        self.clip_norm = 3

        # for early stop
        self.lr_patience = 3
        self.decay_patience = 2

        self.pre_trained = True
        self.data_path = self.root_path + "/data/examples.pkl"
        self.model_path = self.root_path + "/dumps/disco_model.pt"

        # max length
        self.if_C = True
        self.C = 6



    def __repr__(self):
        return str(vars(self))


config = Config()
