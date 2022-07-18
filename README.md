<p align="center">
  <img src="docs/logo.png" alt="pytorch template logo">
</p>
English | [简体中文](./README-CN.md)

# PyTorch Template
PyTorch deep learning project made easy and accelerated.
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->
* [PyTorch Template](#pytorch-template-project)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
    * [Multi-GPUs/TPU/fp16](#Distributed-Data-Parallel-and-Mixed-Precision-Training)
	* [Customization](#customization)
		* [Custom CLI options](#custom-cli-options)
	* [License](#license)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Features
* Clear folder structure which is suitable for many deep learning projects.
* `.json` config file support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Accelerate the process of training PyTorch models by providing a minimal, but extensible training loop:
  * A simple and contained, but easily customisable, training loop, which should work out of the box in straightforward cases; behaviour can be customised using inheritance and/or callbacks.
  * Handles device placement, mixed-precision, DeepSpeed integration, multi-GPU and distributed training with no code changes.
  * Uses pure PyTorch components, with no additional modifications or wrappers, and easily interoperates with other popular libraries such as timm, transformers and torchmetrics.

## Folder Structure
  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── config.json - holds configuration for training
  ├── config4test.json - holds configuration for testing
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── dataset/ - anything about dataset goes here
  │   └── datasets.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── pytorch_accelerated/ - trainers, currently just copy the pytorch_accelerated library here since its APIs are not stable now   
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

## Usage
### Install depredencies
```py
pip install -r requirements
```
### Run
The code in this repo is an MNIST example of the template.
Try `python train.py -c config.json` to run code.

### Config file format
Config files are in `.json` format:
```javascript
{
  "name": "Mnist_LeNet",        // training session name
  
  "arch": {
    "type": "MnistModel",       // name of model architecture to train
    "args": {

    }                
  },
  "dataset": {
    "type": "MnistDataset",      // selecting dataset
    "args":{
        "data_dir": "data/",     // dataset path
        "validation_split": 0.1  // size of validation dataset. float(portion)
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                     // learning rate
      "weight_decay": 0,               // (optional) weight decay
      "amsgrad": true
    }
  },
  "loss": "nll_loss",                  // loss
  "metrics": [
    "accuracy"                         // list of metrics to evaluate
  ],                         
  "lr_scheduler": {
    "type": "StepLR",                  // learning rate scheduler
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 100,                     // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "train_dataloader_args": {         // dataloader arguments for training
       "batch_size": 128,              // batch size
       "shuffle": true,                // shuffle training data before splitting
       "num_workers": 2                // number of cpu processes to be used for data loading
    },
    "eval_dataloader_args": {          // dataloader arguments for evaluation
       "batch_size": 128, 
       "shuffle": false,
       "num_workers": 1
    }
  }
}
```

Add addional configurations if you need.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Distributed Data Parallel and Mixed Precision Training
If you want the powerful multi-GPUs/TPU/fp16 training mode, just use the CLI provided by [accelerate](https://github.com/huggingface/accelerate):
```sh
accelerate config --config_file accelerate_config.yaml
```
and answer the questions asked. Then:
```sh
accelerate launch --config_file accelerate_config.yaml train.py -c config.json
```


## Customization

### Project initialization
Use the `new_project.py` script to make your new project directory with template files.
`python new_project.py ../NewProject` then a new project folder named 'NewProject' will be made.
This script will filter out unneccessary files like cache, git files or readme file. 

### Custom CLI options

Changing values of config file is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.

This template uses the configurations stored in the json file by default, but by registering custom options as follows
you can change some of them using CLI flags.

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```
`target` argument should be sequence of keys, which are used to access that option in the config dict. In this example, `target` 
for the learning rate option is `('optimizer', 'args', 'lr')` because `config['optimizer']['args']['lr']` points to the learning rate.
`python train.py -c config.json --bs 256` runs training with options given in `config.json` except for the `batch size`
which is increased to 256 by command line options.

### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

## License
This project is licensed under the MIT License. See  LICENSE for more details.

## Acknowledgements
This project is inspired and empowered by two projects: [pytorch-template](https://github.com/victoresque/pytorch-template) by [@victoresque](https://github.com/victoresque) and [pytorch-accelerated](https://github.com/Chris-hughes10/pytorch-accelerated) by [@Chris-hughes10](https://github.com/Chris-hughes10).