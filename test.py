import sys
sys.path.extend(["/opt/public/airlock/lihang/pyclipper/", "/opt/public/airlock/lihang/Polygon3/", "/opt/public/airlock/lihang/deepvac/"])
from deepvac import Deepvac, LOG
from modules.cpp import pse
import torch
import time
import cv2
import os
import numpy as np

class DeepvacPseTest(Deepvac):
    def __init__(self, deepvac_config):
        super(DeepvacPseTest,self).__init__(deepvac_config)

    def testFly(self):
        for idx, (org_img, img) in enumerate(self.config.test_loader):
            LOG.logI('progress: %d / %d'%(idx+1, len(self.config.test_loader)))
            org_img = org_img.numpy().astype('uint8')[0]

            img = img.to(self.config.device)
            start_time = time.time()
            outputs = self.config.net(img)
            print(time.time()-start_time)

            score = torch.sigmoid(outputs[:, 0, :, :])
            outputs = (torch.sign(outputs - self.config.binary_th) + 1) / 2

            text = outputs[:, 0, :, :]
            kernels = outputs[:, 0:self.config.kernel_num, :, :] * text

            score = score.data.cpu().numpy()[0].astype(np.float32)
            text = text.data.cpu().numpy()[0].astype(np.uint8)
            kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)

            # c++ version pse
            pred = pse(kernels, self.config.min_kernel_area / (self.config.scale * self.config.scale))

            scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
            label = pred
            label_num = np.max(label) + 1
            bboxes = []
            for i in range(1, label_num):
                points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

                if points.shape[0] < self.conf.test.min_area / (self.config.scale * self.config.scale):
                    continue

                score_i = np.mean(score[label == i])
                if score_i < self.config.min_score:
                    continue

                rect = cv2.minAreaRect(points)
                crop_box = cv2.boxPoints(rect)
                crop_box *= scale
                crop_box[ :, 0] = np.clip(crop_box[ :, 0], 0, org_img.shape[1])
                crop_box[ :, 1] = np.clip(crop_box[ :, 1], 0, org_img.shape[0])
                x_max, y_max = np.max(crop_box, axis=0)
                x_min, y_min = np.min(crop_box, axis=0)
                org_img = cv2.rectangle(org_img, (x_min, y_min), (x_max, y_max), (255,0,0), 2)

            cv2.imwrite(os.path.join(self.config.output_dir ,str(idx).zfill(3)+'.jpg'), org_img)
        self.config.sample = img

if __name__ == '__main__':
    from config import config as deepvac_config
    Pse = DeepvacPseTest(deepvac_config)
    Pse()
