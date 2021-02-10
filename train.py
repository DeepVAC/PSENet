import sys
from deepvac import DeepvacTrain, LOG, is_ddp 
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.autograd import Variable

from modules.model_mv3fpn import FpnMobileNetv3
from modules.utils import preprocess, dice_loss, ohem_batch, cal_text_score
from modules.utils_metrics import runningScore
from aug.aug import PseTrainDataset

import cv2
import os
import time
import numpy as np

class DeepvacPse(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(DeepvacPse,self).__init__(deepvac_config)

    def initNetWithCode(self):
        self.net = FpnMobileNetv3(kernel_num=self.conf.train.kernel_num)
        self.net.to(self.device)

    def initOptimizer(self):
        self.initAdamOptimizer()

    def initCriterion(self):
        self.criterion = dice_loss

    def initTrainLoader(self):
        self.train_dataset = PseTrainDataset(self.conf.train)
        if is_ddp:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(
            dataset = self.train_dataset,
            batch_size=self.conf.train.batch_size,
            shuffle=False if is_ddp else self.conf.train.shuffle,
            num_workers=self.conf.workers,
            drop_last=self.conf.drop_last,
            pin_memory=self.conf.pin_memory,
            sampler=self.train_sampler if is_ddp else None
        )

    def initValLoader(self):
        self.val_dataset = PseTrainDataset(self.conf.val)
        self.val_loader = DataLoader(
            dataset = self.val_dataset,
            batch_size=self.conf.val.batch_size,
            shuffle=self.conf.val.shuffle,
            num_workers=self.conf.workers,
            drop_last=self.conf.drop_last,
            pin_memory=self.conf.pin_memory,
        )

    def initTestLoader(self):
        pass

    def preIter(self):
        self.gt_texts, self.gt_kernels, self.training_masks = self.target

    def earlyIter(self):
        start = time.time()
        self.sample = self.sample.to(self.device)
        self.gt_texts = self.gt_texts.to(self.device)
        self.gt_kernels = self.gt_kernels.to(self.device)
        self.training_masks = self.training_masks.to(self.device)
        if not self.is_train:
            return

        self.data_cpu2gpu_time.update(time.time() - start)
        try:
            self.addGraph(self.sample)
        except:
            LOG.logW("Tensorboard addGraph failed. You network foward may have more than one parameters?")
            LOG.logW("Seems you need reimplement preIter function.")

    def doForward(self):
        self.outputs = self.net(self.sample)

    def doLoss(self):
        self.texts = self.outputs[:, 0, :, :]
        kernels = self.outputs[:, 1:, :, :]

        selected_masks = ohem_batch(self.texts, self.gt_texts, self.training_masks)
        selected_masks = Variable(selected_masks.cuda())

        loss_text = self.criterion(self.texts, self.gt_texts, selected_masks)

        loss_kernels = []
        mask0 = torch.sigmoid(self.texts).data.cpu().numpy()
        mask1 = self.training_masks.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        selected_masks = Variable(selected_masks.cuda())
        for i in range(6):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = self.gt_kernels[:, i, :, :]
            loss_kernel_i = self.criterion(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernel = sum(loss_kernels) / len(loss_kernels)

        self.loss = 0.7 * loss_text + 0.3 * loss_kernel

    def postIter(self):
        if self.is_train:
            return

        self.score_text = cal_text_score(self.texts, self.gt_texts, self.training_masks, self.running_metric_text)

    def preEpoch(self):
        if self.is_train:
            return 
        self.running_metric_text = runningScore(2)

    def postEpoch(self):
        if self.is_train:
            return

        self.accuracy = self.score_text['Mean Acc']
        LOG.logI('Test accuray: {:.4f}'.format(self.accuracy))

    def processAccept(self):
        pass



if __name__ == '__main__':
    from config import config as deepvac_config
    Pse = DeepvacPse(deepvac_config)
    Pse()
