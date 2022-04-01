# -----------------------------------------------------------
# Multi-Perspective Network implementation based on
# "Multi-Perspective Motivated Neural Network for Image-Text Matching", and part of the code refer to SCAN
# Xueyang Qin, Lishuang Li, Guangyao Pang
#
# Writen by Xueyang Qin, 2020
# ---------------------------------------------------------------
"""Training script"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7,6'
import time
import shutil

import torch
torch.backends.cudnn.enabled = False
import numpy
import data
from vocab import Vocabulary, deserialize_vocab
from model import MPMNN, sim_i2t, sim_t2i, global_cos
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, \
    shard_xattn_t2i, shard_xattn_i2t, glo_shard_xattn_t2i, glo_shard_xattn_i2t, sal_shard_xattn_i2t, sal_shard_xattn_t2i
from torch.autograd import Variable
import logging
import tensorboard_logger as tb_logger
from data_parallel_my import BalancedDataParallel
from Sim_Fusion import SimFusion
import argparse
def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=20, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=150, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the local joint embedding.')
    parser.add_argument('--global_embed_size', default=2048, type=int,
                        help='Dimensionality of the global embedding size.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=50, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=10000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='./runs/runX/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/runX/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_false',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_false',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--raw_feature_norm', default="clipped_l2norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--agg_func', default="LogSumExp",
                        help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--cross_attn', default="t2i",
                        help='t2i|i2t')
    parser.add_argument('--precomp_enc_type', default="basic",
                        help='basic|weight_norm')
    parser.add_argument('--bi_gru', action='store_false',
                        help='Use bidirectional GRU.')
    parser.add_argument('--lambda_lse', default=6., type=float,
                        help='LogSumExp temp.')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                        help='Attention softmax temperature.')
    parser.add_argument('--gpu0_size', default=32, type=int,
                        help='Batch_size of gpu0.')
    parser.add_argument('--Channel', default=1, type=int,
                        help='Numbers of Channel.')
    parser.add_argument('--kernel_num', default=1, type=int,
                        help='Numbers of conv kernel.')
    parser.add_argument('--kernel_size', default=[3, 5, 7, 9, 11], type=list,
                        help='Size of conv kernel.')
    parser.add_argument('--salience_embed_size', default=4096, type=int,
                        help='Dimensionality of the salience embedding size.')
    parser.add_argument('--salience_img_dim', default=2048, type=int,
                        help='Dimensionality of the salience embedding size.')
    parser.add_argument('--p', default=0.001, type=float,
                        help='Probability of the dropout.')
    parser.add_argument('--phi', default=0.2, type=float,
                        help='Threshold for similarity filtering.')

    opt = parser.parse_args()
    print(opt)


    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # Load data loaders
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = MPMNN(opt)
    model = BalancedDataParallel(opt.gpu0_size, model, dim=0).cuda()

    best_rsum = 0
    start_epoch = 0
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_rsum = checkpoint['best_rsum']
            model.module.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.module.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        print(opt.logger_name)
        print(opt.model_name)

        adjust_learning_rate(opt, model.module.optimizer, epoch)

        train(opt, train_loader, model, epoch, val_loader)

        rsum = validate(opt, val_loader, model)
        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch,
            'model': model.module.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.module.Eiters,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.module.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.module.logger = train_logger

        # Update the model
        model.module.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.module.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.module.logger)))

        # validate at every val_step
        if model.module.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model)


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, cap_lens, glo_img, glo_txt, sal_img, sal_txt = encode_data(
        model, val_loader, opt.log_step, logging.info)

    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    glo_img = numpy.array([glo_img[i] for i in range(0, len(glo_img), 5)])
    sal_img = numpy.array([sal_img[i] for i in range(0, len(sal_img), 5)])

    start = time.time()
    if opt.cross_attn == 't2i':
        sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        global_sims = glo_shard_xattn_t2i(glo_img, glo_txt, cap_lens, opt, shard_size=128)
        salience_sims = sal_shard_xattn_t2i(sal_img, sal_txt, cap_lens, opt, shard_size=128)


    elif opt.cross_attn == 'i2t':
        sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        global_sims = glo_shard_xattn_i2t(glo_img, glo_txt, cap_lens, opt, shard_size=128)
        salience_sims = sal_shard_xattn_i2t(sal_img, sal_txt, cap_lens, opt, shard_size=128)
    else:
        raise NotImplementedError
    end = time.time()
    print("calculate similarity time:", end-start)
    simfusion = SimFusion(opt.phi)
    Co_sim = simfusion(sims, global_sims, salience_sims)
    Co_sim = (sims + global_sims + salience_sims)/3.0
    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, Co_sim)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs, cap_lens, Co_sim)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):

    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
