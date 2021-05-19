import torch
import torch.optim as optim

from deepvac import config, AttrDict

from data.dataloader import PseTrainDataset, PseTestDataset
from modules.model_mv3fpn import FpnMobileNetv3
from modules.loss import PSELoss

## ------------------ common ------------------
config.core.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.output_dir = 'output'
config.core.log_every = 100
config.core.disable_git = True
config.core.model_reinterpret_cast = True
config.core.cast_state_dict_strict = False
#config.core.jit_model_path = "./output/script.pt"

## -------------------- training ------------------
## train runtime
config.core.epoch_num = 200
config.core.save_num = 1

## -------------------- tensorboard ------------------
#config.core.tensorboard_port = "6007"
#config.core.tensorboard_ip = None

## -------------------- script and quantize ------------------
#config.cast.script_model_dir = "./output/script.pt"

## -------------------- net and criterion ------------------
config.core.net = FpnMobileNetv3(kernel_num=7)
config.core.criterion = PSELoss(config)

## -------------------- optimizer and scheduler ------------------
config.core.optimizer = torch.optim.Adam(config.core.net.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

lambda_lr = lambda epoch: round ((1 - epoch/config.core.epoch_num) ** 0.9, 8)
config.core.scheduler = optim.lr_scheduler.LambdaLR(config.core.optimizer, lr_lambda=lambda_lr)

## -------------------- loader ------------------
sample_path = 'your train image dir'
label_path = 'your train labels dir'
is_transform = True
img_size = 640
config.datasets.PseTrainDataset = AttrDict()
config.datasets.PseTrainDataset.kernel_num = 7
config.datasets.PseTrainDataset.min_scale = 0.4
config.core.train_dataset = PseTrainDataset(config, sample_path, label_path, is_transform, img_size)
config.core.train_loader = torch.utils.data.DataLoader(
    dataset = config.core.train_dataset,
    batch_size = 12,
    shuffle = True,
    num_workers = 4,
    pin_memory = True,
    sampler = None
)

## -------------------- val ------------------
sample_path = 'your val image dir'
label_path = 'your val labels dir'
is_transform = True
img_size = 640
config.core.val_dataset = PseTrainDataset(config, sample_path, label_path, is_transform, img_size)
config.core.val_loader = torch.utils.data.DataLoader(
    dataset = config.core.val_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
    pin_memory = True
)


## -------------------- test ------------------
config.core.model_path = "your test model dir / pretrained weights"
config.core.kernel_num = 7
config.core.min_kernel_area = 10.0
config.core.min_area = 300.0
config.core.min_score = 0.93
config.core.binary_th = 1.0
config.core.scale = 1

sample_path = 'your test image dir'
config.core.test_dataset = PseTestDataset(config, sample_path, long_size=1280)
config.core.test_loader = torch.utils.data.DataLoader(
    dataset = config.core.test_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
    pin_memory = True
)

## ------------------------- DDP ------------------
config.dist_url = 'tcp://172.16.90.55:27030'
config.world_size = 1
