#!/usr/bin/env python
import numpy as np
import pickle
import random
from random import shuffle
from training.util import adjust_learning_rate, clip_model_grad, create_opt, sample2tensor, set_seed
from util.evaluate import get_f1, get_f1_coarse
from model.Coarse2Fine import SemiEnt
from config import config
from torch.autograd import Variable
import torch
import copy
import time
import pdb

# load data
with open("./data/examples.pkl", "rb") as f:
    train_examples, dev_examples, test_examples = pickle.load(f)

with open("./data/word_vec_{}.pkl".format(config.token_embed), "rb") as f:
    id2word, word2id, id2char, char2id, word2vec = pickle.load(f)

config.voc_size = len(id2word)
config.char_voc_size = len(id2char)
set_seed(config.seed)

# ner_model = Disco(config)
ner_model = SemiEnt(config)
if config.pre_trained:
    ner_model.load_vector(word2vec)
if config.if_gpu and torch.cuda.is_available(): ner_model = ner_model.cuda()

parameters = filter(lambda p: p.requires_grad, ner_model.parameters())
optimizer = create_opt(parameters, config)

if config.if_shuffle: shuffle(train_examples)
print(str(config))

# Test
# f1 = get_f1_coarse(ner_model, dev_examples)

train_start_time = time.time()
early_counter = 0
decay_counter = 0
best_per = 0
for e_ in range(config.epoch):
    print("Epoch: ", e_ + 1)
    batch_counter = 0
    for ie, example in enumerate(train_examples):
        token_ids, char_ids, entities, class_samples = example

        # skip for initial experiments
        # if len(token_ids) > 20:
        #    continue

        token_var = Variable(torch.LongTensor(np.array(token_ids)))
        sample_vars = sample2tensor(class_samples, config.if_gpu)
        if config.if_gpu:  token_var = token_var.cuda()

        char_vars = []
        for char_id_l in char_ids:
            char_var = Variable(torch.LongTensor(np.array(char_id_l)))
            if config.if_gpu: char_var = char_var.cuda()
            char_vars.append(char_var)

        ner_model.train()
        optimizer.zero_grad()
        loss = ner_model.forward(token_var, char_vars, entities, sample_vars)
        loss.backward()
        clip_model_grad(ner_model, config.clip_norm)
        print("{2}: sentece length {0} : loss {1}".format(len(token_ids), loss.item(), ie))
        batch_counter += 1

        optimizer.step()

    if (e_+1) % config.check_every != 0:
        continue

    # evaluating dev and always save the best
    cur_time = time.time()
    f1_coarse = get_f1_coarse(ner_model, dev_examples)
    f1 = get_f1(ner_model, dev_examples)
    print("Dev step took {} seconds".format(time.time() - cur_time))

    # early stop
    if f1 > best_per:
        early_counter = 0
        best_per = f1
        # del best_model
        del ner_model.hypergraph.span_cache # non-leaf nodes
        best_model = copy.deepcopy(ner_model)
    else:
        early_counter += 1
        if early_counter > config.lr_patience:
            decay_counter += 1
            early_counter = 0
            if decay_counter > config.decay_patience:
                break
            else:
                adjust_learning_rate(optimizer)
print("")
print("Training step took {} seconds".format(time.time() - train_start_time))
print("Best dev acc {0}".format(best_per))
print("")

# remember to eval after loading the model. for the reason of batchnorm and dropout
cur_time = time.time()
f1_dev = get_f1(best_model, dev_examples) # sanity check
f1_test_coarse = get_f1_coarse(best_model, test_examples)
f1_test = get_f1(best_model, test_examples)
print("Test step took {} seconds".format(time.time() - cur_time))

serial_number = str(random.randint(0,248))
this_model_path = config.model_path + "_" + serial_number
print("Dumping model to {0}".format(this_model_path))
# torch.save(best_model.state_dict(), this_model_path)
