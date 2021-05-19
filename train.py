import sys
sys.path.extend(["/opt/public/airlock/lihang/pyclipper/", "/opt/public/airlock/lihang/Polygon3/", "/opt/public/airlock/lihang/deepvac/"])
from deepvac import DeepvacTrain, LOG 
from modules.utils import cal_text_score, runningScore 

class PSENetTrain(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(PSENetTrain,self).__init__(deepvac_config)

    def doFeedData2Device(self):
        self.config.sample = self.config.sample.to(self.config.device)
        if self.config.target is not None:
            self.config.target = [tar.to(self.config.device) for tar in self.config.target]

    def doLoss(self):
        if not self.config.is_train:
            return
        self.config.loss = self.config.criterion(self.config.output, self.config.target)

    def postIter(self):
        if self.config.is_train:
            return
        
        self.score_text = cal_text_score(self.config.output[:, 0, :, :], self.config.target[0], self.config.target[2], self.running_metric_text)

    def preEpoch(self):
        if self.config.is_train:
            return 
        self.running_metric_text = runningScore(2)

    def postEpoch(self):
        if self.config.is_train:
            return

        self.accuracy = self.score_text['Mean Acc']
        LOG.logI('Test accuray: {:.4f}'.format(self.accuracy))

if __name__ == '__main__':
    from config import config as deepvac_config
    Pse = PSENetTrain(deepvac_config)
    Pse()
