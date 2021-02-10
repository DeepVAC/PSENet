from deepvac.syszux_config import *
# DDP
config.dist_url = 'tcp://172.16.90.55:27030'
config.world_size = 1

config.disable_git = True
config.workers = 3
config.device = 'cuda'
config.epoch_num = 200
config.lr = 0.001
config.lr_step = [50, 100]
config.lr_factor = 0.1
config.save_num = 1
config.log_every = 10
config.momentum = 0.99
config.weight_decay=5e-4
config.nesterov = False
config.drop_last = True
config.pin_memory = True

config.output_dir = 'output'
#config.trace_model_dir = './trace.pt'
#config.script_model_dir = 'output/script.pt'
#config.static_quantize_dir = "./trace.pt.sq"
#config.jit_model_path = "./trace.pt"
# train
config.train.data_dir = 'ctw_root_dir/train/text_image/'
config.train.gt_dir = 'ctw_root_dir/train/text_label_curve/'
config.train.batch_size = 12
config.train.shuffle = True
config.train.img_size = 640
config.train.is_transform = True
config.train.kernel_num = 7
config.train.min_scale = 0.4
config.train.arch = 'mv3'

# val
config.val.data_dir = 'ctw_root_dir/test/text_image/'
config.val.gt_dir = 'ctw_root_dir/test/text_label_curve/'
config.val.batch_size = 1
config.val.shuffle = False
config.val.img_size = 640
config.val.is_transform = True
config.val.kernel_num = 7
config.val.min_scale = 0.4

#test
config.model_path = '<your model path>'
config.test.fileline_data_path_prefix = <test-data-path>
config.test.fileline_path = <image-name-to-coordinate-maps>
config.test.batch_size = 1
config.test.shuffle = False
config.test.arch = 'mv3'
config.test.long_size = 1280
config.test.kernel_num = 7
config.test.min_kernel_area = 10.0
config.test.min_area = 300.0
config.test.min_score = 0.93
config.test.binary_th = 1.0
config.test.scale = 1
config.test.use_fileline = True
