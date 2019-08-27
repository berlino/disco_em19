from collections import defaultdict
from torch.autograd import Variable
from training.util import sample2tensor
import numpy as np
import pdb
import torch

def evaluate(gold_entities, pred_entities):
    prec_all_num, prec_num, recall_all_num, recall_num = 0, 0, 0, 0
    for entity in pred_entities:
        if entity in gold_entities:
            prec_num += 1

    for entity in gold_entities:
        if entity in pred_entities:
            recall_num += 1

    recall_all_num += len(pred_entities)
    prec_all_num += len(gold_entities)

    return prec_all_num, prec_num, recall_all_num, recall_num

def get_f1(model, examples):
    model.eval()
    pred_all, pred, recall_all, recall = 0, 0, 0, 0
    f_pred_all, f_pred, f_recall_all, f_recall = 0, 0, 0, 0

    for example in examples:
        token_ids, char_ids, entities, _ = example
        token_var = Variable(torch.LongTensor(np.array(token_ids)))
        if next(model.parameters()).is_cuda:
            token_var = token_var.cuda()

        char_vars = []
        for char_id_l in char_ids:
            char_var = Variable(torch.LongTensor(np.array(char_id_l)))
            if next(model.parameters()).is_cuda: char_var = char_var.cuda()
            char_vars.append(char_var)

        pred_entities = model.predict(token_var, char_vars)
        p_a, p, r_a, r = evaluate(entities, pred_entities)

        pred_all += p_a
        pred += p
        recall_all += r_a
        recall += r


    print(pred_all, pred, recall_all, recall)
    if recall == 0 or pred == 0:
        f1 = 0
    else:
        f1 = 2 / ((pred_all / pred) + (recall_all / recall))
    print( "Precision {0}, Recall {1}, F1 {2}".format(pred / pred_all, recall / recall_all, f1) )
    # print("Prediction Crossing: ", pred_cross_num)
    # print("Gold Crossing: ", gold_cross_num)

    return f1

def get_f1_coarse(model, examples):
    model.eval()
    pred_all, pred, recall_all, recall = 0, 0, 0, 0
    f_pred_all, f_pred, f_recall_all, f_recall = 0, 0, 0, 0

    for example in examples:
        token_ids, char_ids, entities, _ = example
        token_var = Variable(torch.LongTensor(np.array(token_ids)))

        if next(model.parameters()).is_cuda:
            token_var = token_var.cuda()

        char_vars = []
        for char_id_l in char_ids:
            char_var = Variable(torch.LongTensor(np.array(char_id_l)))
            if next(model.parameters()).is_cuda: char_var = char_var.cuda()
            char_vars.append(char_var)

        segs = model.disco2coarse(entities)
        pred_segs = model.predict_coarse(token_var, char_vars)
        p_a, p, r_a, r = evaluate(segs, pred_segs)

        pred_all += p_a
        pred += p
        recall_all += r_a
        recall += r


    print(pred_all, pred, recall_all, recall)
    f1 = 2 / ((pred_all / pred) + (recall_all / recall))
    print( "Evaluating coarse segments: Precision {0}, Recall {1}, F1 {2}".format(pred / pred_all, recall / recall_all, f1) )
    # print("Prediction Crossing: ", pred_cross_num)
    # print("Gold Crossing: ", gold_cross_num)

    return f1


if __name__ == "__main__":
    num = evaluate([[(0,2,2), (1,3,1)]], [])
    print(num)
