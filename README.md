# UOOD-MCD
This is an pytorch implementation of [Unsupervised Out-of-Distribution Detection by Maximum Classifier Discrepancy](https://arxiv.org/pdf/1908.04951.pdf). 

## Requirements
- Python 3.7
- PyTorch 1.1.0
- torchvision 0.3.0
- progress
- matplotlib
- numpy

## Preparation
Download five out-of-distributin datasets provided by [ODIN](https://github.com/ShiyuLiang/odin-pytorch):

* **[Tiny-ImageNet (crop)](https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz)**
* **[Tiny-ImageNet (resize)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)**
* **[LSUN (crop)](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)** 
* **[LSUN (resize)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)** 
* **[iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)**

Here is an example code of downloading Tiny-ImageNet (crop) dataset. In the **root** directory, run

```
mkdir data
cd data
wget https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz
tar -xvzf Imagenet.tar.gz
cd ..
```

## Usage

### Download Pre-trained Models on CIFAR
We provide download links of cifar10/100 pre-trained models.
* **[Pretrained Models](https://www.dropbox.com/s/qjitycxijexzp8y/pretrained.zip)** 

In the **root** directory, run

```
mkdir pretrained
cd pretrained
wget https://www.dropbox.com/s/qjitycxijexzp8y/pretrained.zip
unzip pretrained.zip
cd ..
```

### Train single model
Finetune DenseNet on CIFAR-10 as ID and TinyImageNet as OOD.
```
python train.py -c checkpoints/cifar10_Imagenet_ckpt --gpu 0 --resume pretrained/cifar10_dense.pth.tar --out-dataset Imagenet
```
Trained model will be saved at `checkpoints/cifar10_Imagenet_ckpt`.

### Train all models
```
python train_all.py --gpu 0
```
This script will finetune models of DenseNet/WideResNet on CIFAR-10/100 as ID and five other datasets as OOD which results in 20 models.
Trained model will be saved at `checkpoints`.

### Test
For example, to test DenseNet-BC trained on CIFAR-10 where TinyImageNet (crop) is the out-of-distribution dataset, please run 
```
python test.py --result checkpoints/cifar10_Imagenet_ckpt
```


## References
- [1]: Q. Yu and K. Aizawa. "Unsupervised Out-of-Distribution Detection by Maximum Classifier Discrepancy", in ICCV, 2019.
- [2]: S. Liang, Y. Li and R. Srikant. "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks", in ICLR, 2018.
