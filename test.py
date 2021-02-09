import sys
from deepvac import Deepvac, FileLineCvStrDataset, LOG
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

from modules.cpp import pse
from modules.model_mv3fpn import mobilenetv3
from modules.utils import preprocess
import time
import cv2
import os
import numpy as np

class PseDataset(FileLineCvStrDataset):
    def __init__(self, config):
        super(PseDataset, self).__init__(config)
        self.long_size = config.long_size
    
    def __getitem__(self, idx):
        img, target = super(PseDataset, self).__getitem__(idx)
        return preprocess(img, self.long_size, target)

class DeepvacPseTest(Deepvac):
    def __init__(self, deepvac_config):
        super(DeepvacPseTest,self).__init__(deepvac_config)
        if len(sys.argv) != 1:
            assert len(sys.argv)==2, 'You can only pass a image path !'
            LOG.logI('Find image: {}'.format(sys.argv[1]))
            self.conf.test.use_fileline = False
            self.conf.test.image_path = sys.argv[1]
        self.initTestLoader()

    def initNetWithCode(self):
        self.net = mobilenetv3(kernel_num=self.conf.test.kernel_num)
        self.net.to(self.device)

    def report(self):
        for idx, (org_img, img, labels) in enumerate(self.test_loader):
            LOG.logI('progress: %d / %d'%(idx, len(self.test_loader)))
            org_img = org_img.numpy().astype('uint8')[0]
            org_img = org_img.copy()

            img = img.to(self.device)
            start_time = time.time()
            outputs = self.net(img)
            print(time.time()-start_time)

            score = torch.sigmoid(outputs[:, 0, :, :])
            outputs = (torch.sign(outputs - self.conf.test.binary_th) + 1) / 2

            text = outputs[:, 0, :, :]
            kernels = outputs[:, 0:self.conf.test.kernel_num, :, :] * text

            score = score.data.cpu().numpy()[0].astype(np.float32)
            text = text.data.cpu().numpy()[0].astype(np.uint8)
            kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)

            # c++ version pse
            pred = pse(kernels, self.conf.test.min_kernel_area / (self.conf.test.scale * self.conf.test.scale))

            scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
            label = pred
            label_num = np.max(label) + 1
            bboxes = []
            for i in range(1, label_num):
                points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

                if points.shape[0] < self.conf.test.min_area / (self.conf.test.scale * self.conf.test.scale):
                    continue

                score_i = np.mean(score[label == i])
                if score_i < self.conf.test.min_score:
                    continue

                rect = cv2.minAreaRect(points)
                crop_box = cv2.boxPoints(rect)
                crop_box *= scale
                crop_box[ :, 0] = np.clip(crop_box[ :, 0], 0, org_img.shape[1])
                crop_box[ :, 1] = np.clip(crop_box[ :, 1], 0, org_img.shape[0])
                x_max, y_max = np.max(crop_box, axis=0)
                x_min, y_min = np.min(crop_box, axis=0)
                org_img = cv2.rectangle(org_img, (x_min, y_min), (x_max, y_max), (255,0,0), 2)

            cv2.imwrite('output/vis/'+str(idx).zfill(3)+'.jpg', org_img)


    def process(self):
        self.report()

    def initTestLoader(self):
        if self.conf.test.use_fileline:
            self.test_dataset = PseDataset(self.conf.test)
            self.test_loader = DataLoader(
                dataset=self.test_dataset,
                batch_size=self.conf.test.batch_size,
                shuffle=self.conf.test.shuffle,
                num_workers=self.conf.workers,
                drop_last=True
            )
        else:
            self.test_loader = preprocess(cv2.imread(self.conf.test.image_path), self.conf.test.long_size)


if __name__ == '__main__':
    from config import config as deepvac_config
    Pse = DeepvacPseTest(deepvac_config)
    input_tensor = torch.rand(1,3,640,640)
    Pse(input_tensor)
