# PSENet
DeepVAC-compliant PSENet implementation.

# 简介
本项目实现了符合DeepVAC规范的OCR检测模型PSENet

**项目依赖**

- deepvac >= 0.3.1
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
config.train.data_dir = <your train image path>
config.train.gt_dir = <your train gt path>
config.val.data_dir = <your val image path>
config.val.gt_dir = <your val gt path>
```

## 4. 训练相关配置

- dataloader相关配置(config.train, config.val)

```python
config.train.batch_size = 12
config.train.shuffle = True
config.train.img_size = 640
config.train.is_transform = True    # 是否动态数据增强
config.train.kernel_num = 7         # PSE kernel数量
config.train.min_scale = 0.4        # 最小kernel的缩放倍数
config.train.arch = 'mv3'           # 网络backbone,目前只支持mv3
```

## 5. 训练

### 5.1 单卡训练

```bash
python3 train.py
```

### 5.2 分布式训练

在config.py中修改如下配置：
```python
#dist_url，单机多卡无需改动，多机训练一定要修改
config.dist_url = "tcp://localhost:27030"

#rank的数量，一定要修改
config.world_size = 2
```
然后执行命令：

```bash
python train.py --rank 0 --gpu 0
python train.py --rank 1 --gpu 1
```

## 6. 测试

- 编译PSE

```python
cd modules/cpp
make
```

- 测试相关配置

```python
config.test.fileline_data_path_prefix = <test-data-path>
config.test.fileline_path = <image-name-to-coordinate-maps>
config.test.batch_size = 1
config.test.shuffle = False
config.test.arch = 'mv3'
config.test.long_size = 1280                        # 输入图片最大边长
config.test.kernel_num = 7                          # PSE kernel数量
config.test.min_kernel_area = 10.0                  # 文本域最小kernel的最低像素数
config.test.min_area = 300.0                        # 文本域名的最低像素数
config.test.min_score = 0.93                        # 判断是文本的最低置信度
config.test.binary_th = 1.0
config.test.scale = 1
```

- 加载模型(*.pth)

```python
config.model_path = '<trained-model-path>'
```

- 运行测试脚本：

```bash
python3 test.py
```

## 7. 使用torchscript模型

如果训练过程中未开启config.script_model_path开关，可以在测试过程中转化torchscript模型
- 转换torchscript模型(*.pt)

```python
config.script_model_path = "output/script.pt"
```
  按照步骤6完成测试，torchscript模型将保存至config.script_model_path指定文件位置

warining: 当前导出torchscript还有问题
```python3
RuntimeError:
Expected a default value of type Tensor (inferred) on parameter "scale".Because "scale" was not annotated with an explicit type it is assumed to be type 'Tensor'.:
```

- 加载torchscript模型

```python
config.jit_model_path = <torchscript-model-path>
```

## 8. 使用静态量化模型
如果训练过程中未开启config.static_quantize_dir开关，可以在测试过程中转化静态量化模型
- 转换静态模型(*.sq)

```python
config.static_quantize_dir = "output/script.sq"
```
  按照步骤6完成测试，静态量化模型将保存至config.static_quantize_dir指定文件位置

- 加载静态量化模型

```python
config.jit_model_path = <static-quantize-model-path>
```

## 9. 更多功能
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

## 10. TODO

- 增加模型导出torchscript
