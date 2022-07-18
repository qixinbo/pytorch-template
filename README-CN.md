<p align="center">
  <img src="docs/logo.png" alt="pytorch template logo">
</p>
English(./README.md) | 简体中文

# PyTorch项目模板
使PyTorch深度学习项目变得简单和高速。
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->
* [PyTorch模板](#PyTorch项目模板)
  * [特性](#特性)
  * [文件结构](#文件结构)
  * [用法](#用法)
    * [配置文件格式](#配置文件格式)
    * [使用配置文件](#使用配置文件)
    * [从检查点恢复](#从检查点恢复)
    * [分布式训练和混合精度训练}](#分布式数据并行和混合精度训练)
  * [自定义](#自定义)
    * [自定义命令行选项](#自定义CLI选项)
  * [许可证](#许可证)
  * [致谢](#致谢)

<!-- /code_chunk_output -->

## 特性
* 清晰的文件夹结构，适用于多种深度学习项目。
* `.json` 配置文件支持便捷的参数调整。
* 可定制的命令行选项，用于更方便地调整参数。
* 检查点保存和恢复。
* 通过提供最小但可扩展的训练循环来加速训练PyTorch模型：
  * 一个简单且包含但易于定制的训练循环，多数情形下开箱即用；可以使用继承和/或回调以自定义行为。
  * 无需更改代码即可处理硬件配置、混合精度、DeepSpeed集成、多GPU和分布式训练。
  * 使用纯PyTorch组件，无需额外修改或包装，并且可以轻松与其他流行的库（如 timm、transformers 和 torchmetrics）进行互操作。

## 文件结构
  ```
  pytorch-template/
  │
  ├── train.py - 开始训练的主要脚本
  ├── test.py - 训练模型的评估
  │
  ├── config.json - 保存训练配置
  ├── config4test.json - 保存测试配置
  ├── parse_config.py - 处理配置文件和 cli 选项的类
  │
  ├── new_project.py - 使用模板文件初始化新项目
  │
  ├── dataset/ - 关于 dataset 的任何信息都在这里
  │ └── datasets.py
  │
  ├── data/ - 存储输入数据的默认目录
  │
  ├── model/ - 模型、损失和指标
  │ ├── model.py
  │ ├── metric.py
  │ └── loss.py
  │
  ├── 保存/
  │ ├── models/ - 训练好的模型保存在这里
  │ └── log/ - tensorboard 和日志输出的默认路径
  │
  ├── pytorch_accelerated/ - 训练器，目前是在此处复制pytorch_accelerated库，因为它的API当前尚不稳定
  │
  ├── logger/ - 用于张量板可视化和日志记录的模块
  │ ├── 可视化.py
  │ ├── logger.py
  │ └── logger_config.json
  │
  └── utils/ - 实用函数
      ├── util.py
      └── ...
  ```

## 用法
### 安装依赖
```py
pip install -r requirements
```
### 运行
当前库中有一个MNIST模板.
使用`python train.py -c config.json`可运行.

### 配置文件格式
配置文件是`.json`格式:
```javascript
{
  "name": "Mnist_LeNet",        // 训练项目的名称
  
  "arch": {
    "type": "MnistModel",       // 模型架构的名称
    "args": {

    }                
  },
  "dataset": {
    "type": "MnistDataset",      // 数据集名称
    "args":{
        "data_dir": "data/",     // 数据集路径
        "validation_split": 0.1  // 验证集的划分比例
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                     // 学习率
      "weight_decay": 0,               // 权重衰减
      "amsgrad": true
    }
  },
  "loss": "nll_loss",                  // 损失
  "metrics": [
    "accuracy"                         // 评估指标
  ],                         
  "lr_scheduler": {
    "type": "StepLR",                  // 学习率调度器
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 100,                     // 训练迭代次数
    "save_dir": "saved/",              // 检查点保存在save_dir/models/name
    "train_dataloader_args": {         // 训练的dataloader所需参数
       "batch_size": 128,              // 批处理大小
       "shuffle": true,                // 是否打乱
       "num_workers": 2                // 数据加载时使用的CPU核数
    },
    "eval_dataloader_args": {          // 验证的dataloader所需参数
       "batch_size": 128, 
       "shuffle": false,
       "num_workers": 1
    }
  }
}
```

如果需要，可添加其他配置。

### 使用配置文件
修改`.json`配置文件中的配置，然后运行：

  ```
  python train.py --config config.json
  ```

### 从检查点恢复
可以通过以下方式从以前保存的检查点恢复：

  ```
  python train.py --resume path/to/checkpoint
  ```

### 分布式数据并行和混合精度训练
如果想要强大的多GPU/TPU/fp16训练模式，只需使用[accelerate](https://github.com/huggingface/accelerate)提供的命令行工具:
```sh
accelerate config --config_file accelerate_config.yaml
```
并回答提出的问题。 然后：
```sh
accelerate launch --config_file accelerate_config.yaml train.py -c config.json
```


## 自定义

### 项目初始化
使用 `new_project.py` 脚本来创建包含模板文件的新项目目录：
```sh
python new_project.py ../NewProject
```
此时将创建一个名为“NewProject”的新项目文件夹。
该脚本将过滤掉不需要的文件，如缓存、git文件或readme文件。 

### 自定义CLI选项
更改配置文件是调整超参数的一种干净、安全且简单的方法。然而，有时需要经常或快速更改某些值，最好有命令行选项。
通过如下注册自定义选项，可以通过命令行来快速更改配置：
  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```
此时`python train.py -c config.json --bs 256`使用`config.json`中给出的选项运行训练，除了`batch size`
是通过命令行选项设置为256。

### 测试
可以通过运行 `test.py` 通过`--resume` 参数传递检查点的路径来测试模型。

## 许可证
该项目在 MIT 许可下获得许可。 有关更多详细信息，请参阅许可证。

## 致谢
该项目主要基于如下两个项目: [pytorch-template](https://github.com/victoresque/pytorch-template) by [@victoresque](https://github.com/victoresque) 和 [pytorch-accelerated](https://github.com/Chris-hughes10/pytorch-accelerated) by [@Chris-hughes10](https://github.com/Chris-hughes10).