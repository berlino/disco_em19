import torch.nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random

def adjust_learning_rate(optimizer):
    cur_lr = optimizer.param_groups[0]['lr']
    # adj_lr = cur_lr / 2
    adj_lr = cur_lr * 0.1
    print("Adjust lr to ", adj_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = adj_lr

def create_opt(parameters, config):
    if config.opt == "SGD":
        optimizer = optim.SGD(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adam":
        optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=config.l2)
    elif config.opt == "Adadelta":
        optimizer = optim.Adadelta(parameters,lr=config.lr, rho=config.rho, eps=config.eps, weight_decay=config.l2)
    elif config.opt == "Adagrad":
        optimizer = optim.Adagrad(parameters, lr=config.lr, weight_decay=config.l2)
    return optimizer


def clip_model_grad(model, clip_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, norm_type=2)

def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic=True

def sample2tensor(samples, if_gpu):
    no_hole, one_hole, two_hole = samples

    no_hole_x1 = [x[0] for x in no_hole[0]]
    no_hole_x2 = [x[1] for x in no_hole[0]]
    no_hole_x1_var = Variable(torch.LongTensor(np.array(no_hole_x1)))
    no_hole_x2_var = Variable(torch.LongTensor(np.array(no_hole_x2)))
    no_hole_y_var = Variable(torch.LongTensor(np.array(no_hole[1])))
    if if_gpu:
        no_hole_x1_var = no_hole_x1_var.cuda()
        no_hole_x2_var = no_hole_x2_var.cuda()
        no_hole_y_var = no_hole_y_var.cuda()
    no_hole_var = ( (no_hole_x1_var, no_hole_x2_var), no_hole_y_var )

    one_hole_x1 = [x[0][0] for x in one_hole[0]]
    one_hole_x2 = [x[0][1] for x in one_hole[0]]
    one_hole_x3 = [x[1][0] for x in one_hole[0]]
    one_hole_x4 = [x[1][1] for x in one_hole[0]]
    one_hole_x1_var = Variable(torch.LongTensor(np.array(one_hole_x1)))
    one_hole_x2_var = Variable(torch.LongTensor(np.array(one_hole_x2)))
    one_hole_x3_var = Variable(torch.LongTensor(np.array(one_hole_x3)))
    one_hole_x4_var = Variable(torch.LongTensor(np.array(one_hole_x4)))
    one_hole_y_var = Variable(torch.LongTensor(np.array(one_hole[1])))
    if if_gpu:
        one_hole_x1_var = one_hole_x1_var.cuda()
        one_hole_x2_var = one_hole_x2_var.cuda()
        one_hole_x3_var = one_hole_x3_var.cuda()
        one_hole_x4_var = one_hole_x4_var.cuda()
        one_hole_y_var = one_hole_y_var.cuda()
    one_hole_var = ( (one_hole_x1_var, one_hole_x2_var, one_hole_x3_var, one_hole_x4_var), one_hole_y_var )

    if len(two_hole[0]) > 0:
        two_hole_x1 = [x[0][0] for x in two_hole[0]]
        two_hole_x2 = [x[0][1] for x in two_hole[0]]
        two_hole_x3 = [x[1][0] for x in two_hole[0]]
        two_hole_x4 = [x[1][1] for x in two_hole[0]]
        two_hole_x5 = [x[2][0] for x in two_hole[0]]
        two_hole_x6 = [x[2][1] for x in two_hole[0]]
        two_hole_x1_var = Variable(torch.LongTensor(np.array(two_hole_x1)))
        two_hole_x2_var = Variable(torch.LongTensor(np.array(two_hole_x2)))
        two_hole_x3_var = Variable(torch.LongTensor(np.array(two_hole_x3)))
        two_hole_x4_var = Variable(torch.LongTensor(np.array(two_hole_x4)))
        two_hole_x5_var = Variable(torch.LongTensor(np.array(two_hole_x5)))
        two_hole_x6_var = Variable(torch.LongTensor(np.array(two_hole_x6)))
        two_hole_y_var = Variable(torch.LongTensor(np.array(two_hole[1])))
        if if_gpu:
            two_hole_x1_var = two_hole_x1_var.cuda()
            two_hole_x2_var = two_hole_x2_var.cuda()
            two_hole_x3_var = two_hole_x3_var.cuda()
            two_hole_x4_var = two_hole_x4_var.cuda()
            two_hole_x5_var = two_hole_x5_var.cuda()
            two_hole_x6_var = two_hole_x6_var.cuda()
            two_hole_y_var = two_hole_y_var.cuda()

        two_hole_var = ((two_hole_x1_var, two_hole_x2_var, two_hole_x3_var, two_hole_x4_var,
            two_hole_x5_var, two_hole_x6_var), two_hole_y_var)
    else:
        two_hole_var = None

    return no_hole_var, one_hole_var, two_hole_var