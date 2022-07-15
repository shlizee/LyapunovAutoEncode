import torch
import argparse
from models import RNNModel
from config import *
from training import *
import time
import pickle as pkl
from generate_trials import *

def main(model_types, task):
    trials = {}
    for i, model_type in enumerate(model_types):
        trials[model_type] = torch.load(f'trials/{task}/{model_type}/{task}_Trials_keep0.2_size64.p')
        val_loss_epochs = trials[model_type].val_loss_epochs
        if i == 0:
            val_losses = val_loss_epochs[10]
        else:
            val_losses = torch.cat((val_losses, val_loss_epochs[10]))
    print(val_losses.shape)
    gt_threshold = torch.median(val_losses)
    gt_mask = val_losses < gt_threshold
    #
    for epoch in [0, 1, 2, 5]:
        for i, model_type in enumerate(model_types):
            val_loss_epochs = trials[model_type].val_loss_epochs
            if i == 0:
                val_loss_epoch = val_loss_epochs[epoch]
            else:
                val_loss_epoch = torch.cat((val_loss_epoch, val_loss_epochs[epoch]))
        epoch_threshold = torch.median(val_loss_epoch)

        epoch_mask = val_loss_epoch <= epoch_threshold
        tp = torch.sum(torch.logical_and(gt_mask, epoch_mask))
        tn = torch.sum(torch.logical_and(~gt_mask, ~epoch_mask))
        fn = torch.sum(torch.logical_and(gt_mask, ~epoch_mask))
        fp = torch.sum(torch.logical_and(~gt_mask, epoch_mask))
        print(gt_threshold, epoch_threshold, tp, tn, fn, fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        if len(model_types) > 1:
            print(f'all: \tepoch: {epoch},\trecall: {recall*100:.1f}, \tprecision: {precision*100:.1f}\t f1 score: {f1:.2f}')
        else:
            print(f'{model_type[0]}: \tepoch: {epoch},\trecall: {recall*100:.1f}, \tprecision: {precision}\t f1 score: {f1:.2f}')
if __name__ == '__main__':
    model_types = ['lstm', 'gru', 'rnn', 'asrnn']
    task = 'SMNIST'
    main(model_types, task)