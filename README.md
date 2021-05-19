# PSENet
DeepVAC-compliant PSENet implementation.

# 简介
本项目实现了符合DeepVAC规范的OCR检测模型PSENet

**项目依赖**

- deepvac >= 0.5.6
- pytorch >= 1.8.0
- torchvision >= 0.7.0
- opencv-python
- numpy
- pyclipper
- Polygon3
- pillow

# 如何运行本项目

## 1. 阅读[DeepVAC规范](https://github.com/DeepVAC/deepvac)
可以粗略阅读，建立起第一印象

## 2. 准备运行环境
使用Deepvac规范指定[Docker镜像](https://github.com/DeepVAC/deepvac#2-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87)

## 3. 准备数据集
- 获取文本检测数据集
  CTW1500格式的数据集,CTW1500下载地址:

  [ch4_training_images.zip](https://rrc.cvc.uab.es/downloads/ch4_training_images.zip)

  [ch4_training_localization_transcription_gt.zip](https://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip)

  [ch4_test_images.zip](https://rrc.cvc.uab.es/downloads/ch4_test_images.zip)

  [Challenge4_Test_Task1_GT.zip](https://rrc.cvc.uab.es/downloads/Challenge4_Test_Task1_GT.zip)


- 数据集配置
  在config.py文件中作如下配置：

``` python
# line 42-43, config for train dataset
sample_path = <your train image path>
label_path = <your train gt path>
# line 60-61, config for val dataset
sample_path = <your val image path>
label_path = <your val gt path>
```

## 4. 训练相关配置

- dataloader相关配置

```python
# line 44-57 for train dataset, line 62-71 for val dataset
is_transform = True                 # 是否动态数据增强
img_size = 640                      # 输入图片大小(img_size, img_size)
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
```

## 5. 训练

```bash
python3 train.py
```
## 6. 测试

- 编译PSE

```bash
cd modules/cpp
make
```

- 测试相关配置

```python
# line 75-91
config.core.model_path = <trained-model-path>
config.core.kernel_num = 7
config.core.min_kernel_area = 10.0
config.core.min_area = 300.0
config.core.min_score = 0.93
config.core.binary_th = 1.0
config.core.scale = 1

sample_path = <your test images path>
config.core.test_dataset = PseTestDataset(config, sample_path, long_size=1280)
config.core.test_loader = torch.utils.data.DataLoader(
    dataset = config.core.test_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
    pin_memory = True
)
```

- 运行测试脚本：

```bash
python3 test.py
```

## 7. 使用torchscript模型

如果训练过程中未开启config.cast.script_model_dir开关，可以在测试过程中转化torchscript模型
- 转换torchscript模型(*.pt)

```python
config.cast.script_model_path = "output/script.pt"
```
按照步骤6完成测试，torchscript模型将保存至config.cast.script_model_dir指定文件位置

- 加载torchscript模型

```python
config.core.jit_model_path = <torchscript-model-path>
```

## 更多功能
如果要在本项目中开启如下功能：
- 预训练模型加载
- checkpoint加载
- 使用tensorboard
- 启用TorchScript
- 转换ONNX
- 转换NCNN
- 转换CoreML
- 开启量化
- 开启自动混合精度训练
- 采用ema策略(config.ema)
- 采用梯度积攒到一定数量在进行反向更新梯度策略(config.nominal_batch_factor)

请参考[DeepVAC](https://github.com/DeepVAC/deepvac)

