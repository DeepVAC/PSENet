import torch
import torch.optim as optim

from deepvac import AttrDict, new

from data.dataloader import PseTrainDataset, PseTestDataset
from modules.model_mv3fpn import FpnMobileNetv3
from modules.loss import PSELoss

config = new('PSENetTrain')
## ------------------ common ------------------
config.core.PSENetTrain.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.PSENetTrain.output_dir = 'output'
config.core.PSENetTrain.log_every = 10
config.core.PSENetTrain.disable_git = True
config.core.PSENetTrain.model_reinterpret_cast = True
config.core.PSENetTrain.cast_state_dict_strict = False
#config.core.PSENetTrain.jit_model_path = "./output/script.pt"

## -------------------- training ------------------
## train runtime
config.core.PSENetTrain.epoch_num = 200
config.core.PSENetTrain.save_num = 1

## -------------------- tensorboard ------------------
# config.core.PSENetTrain.tensorboard_port = "6007"
# config.core.PSENetTrain.tensorboard_ip = None

## -------------------- script and quantize ------------------
config.cast.script_model_dir = "./output/script.pt"

## -------------------- net and criterion ------------------
config.core.PSENetTrain.net = FpnMobileNetv3(kernel_num=7)
config.core.PSENetTrain.criterion = PSELoss(config)

## -------------------- optimizer and scheduler ------------------
config.core.PSENetTrain.optimizer = torch.optim.Adam(config.core.PSENetTrain.net.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

lambda_lr = lambda epoch: round ((1 - epoch/config.core.PSENetTrain.epoch_num) ** 0.9, 8)
config.core.PSENetTrain.scheduler = optim.lr_scheduler.LambdaLR(config.core.PSENetTrain.optimizer, lr_lambda=lambda_lr)

## -------------------- loader ------------------
sample_path = 'your train image dir'
label_path = 'your train labels dir'
is_transform = True
img_size = 640
config.datasets.PseTrainDataset = AttrDict()
config.datasets.PseTrainDataset.kernel_num = 7
config.datasets.PseTrainDataset.min_scale = 0.4
config.core.PSENetTrain.batch_size = 2
config.core.PSENetTrain.num_workers = 4
config.core.PSENetTrain.train_dataset = PseTrainDataset(config, sample_path, label_path, is_transform, img_size)
config.core.PSENetTrain.train_loader = torch.utils.data.DataLoader(
    dataset = config.core.PSENetTrain.train_dataset,
    batch_size = config.core.PSENetTrain.batch_size,
    shuffle = True,
    num_workers = config.core.PSENetTrain.num_workers,
    pin_memory = True,
)

## ------------------------- DDP ------------------
config.core.PSENetTrain.dist_url = 'tcp://localhost:27030'
config.core.PSENetTrain.world_size = 2

## -------------------- val ------------------
sample_path = 'your val image dir'
label_path = 'your val labels dir'
is_transform = True
img_size = 640
config.core.PSENetTrain.val_dataset = PseTrainDataset(config, sample_path, label_path, is_transform, img_size)
config.core.PSENetTrain.val_loader = torch.utils.data.DataLoader(
    dataset = config.core.PSENetTrain.val_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
    pin_memory = True
)

## -------------------- test ------------------
config.core.PSENetTest = config.core.PSENetTrain.clone()
config.core.PSENetTest.model_path = "your test model dir / pretrained weights"
config.core.PSENetTest.kernel_num = 7
config.core.PSENetTest.min_kernel_area = 10.0
config.core.PSENetTest.min_area = 300.0
config.core.PSENetTest.min_score = 0.93
config.core.PSENetTest.binary_th = 1.0
config.core.PSENetTest.scale = 1

sample_path = 'your test image dir'
config.core.PSENetTest.test_dataset = PseTestDataset(config, sample_path, long_size=1280)
config.core.PSENetTest.test_loader = torch.utils.data.DataLoader(
    dataset = config.core.PSENetTest.test_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
    pin_memory = True
)
