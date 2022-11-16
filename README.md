CMGDNet-pytorch

Pytorch implementation paper 'Cross-Modal Guidance Assisted Hierarchical Learning Based Siamese Network for MR Image Denoising'

# Requirements
* Python 3.6 <br>
* Pytorch 1.5.0 <br>
* Torchvision 0.6.1 <br>
* Cuda 10.0

# Usage
This is the Pytorch implementation of CMGDNet. It has been trained and tested on Ubuntu, Cuda 10, Python 3.6 , Pytorch 1.5
and it should also work on windows. 

## To Train 
* Download the pre-trained ImageNet backbone (resnet101 and vgg_conv1), and put it in the 'pretrained' folder
* Download the training dataset and modify the 'train_root' and 'train_list' in the `main.py`. IXI dataset was used in this work; it is publicly available. 
* Set 'mode' to 'train'
* Run `main.py`

## To Test 
* Download the testing dataset and have it in the 'dataset/test/' folder 
* Modify the 'test_folder' in the `main.py` to the testing results saving folder you want
* Set 'mode' to 'test'
* Run `main.py`

## Learning curve
The training log is saved in the 'log' folder. If you want to see the learning curve, you can get it by using: ` tensorboard --logdir your-log-path`

# Pre-trained ImageNet model for training
[resnet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)<br>
[vgg_conv1](https://pan.baidu.com/s/1CJyNALzPIAiHrDSMcRO2yA), password: rllb<br>


# Citation
Please cite our paper if you find the work useful:


@article{naseem2021cross,
  title={Cross-Modal Guidance Assisted Hierarchical Learning Based Siamese Network for MR Image Denoising},
  author={Naseem, Rabia and Alaya Cheikh, Faouzi and Beghdadi, Azeddine and Muhammad, Khan and Sajjad, Muhammad},
  journal={Electronics},
  volume={10},
  number={22},
  pages={2855},
  year={2021},
  publisher={MDPI}
}

