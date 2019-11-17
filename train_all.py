import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

# DenseNet on CIFAR-10
for ood_dataset in ['Imagenet', 'Imagenet_resize', 'LSUN', 'LSUN_resize', 'iSUN']:
    print (f"CIFAR-10 {ood_dataset}")
    subprocess.run(f"python train.py -c checkpoints/cifar10_{ood_dataset}_ckpt --gpu {args.gpu} --resume pretrained/cifar10_dense.pth.tar --out-dataset {ood_dataset}", shell=True)

# DenseNet on CIFAR-100
for ood_dataset in ['Imagenet', 'Imagenet_resize', 'LSUN', 'LSUN_resize', 'iSUN']:
    print (f"CIFAR-100 {ood_dataset}")
    subprocess.run(f"python train.py -c checkpoints/cifar100_{ood_dataset}_ckpt --gpu {args.gpu} --resume pretrained/cifar100_dense.pth.tar --in-dataset cifar100 --out-dataset {ood_dataset}", shell=True)

# WideResNet on CIFAR-10
for ood_dataset in ['Imagenet', 'Imagenet_resize', 'LSUN', 'LSUN_resize', 'iSUN']:
    print (f"CIFAR-10 {ood_dataset}")
    subprocess.run(f"python train.py -c checkpoints/cifar10_{ood_dataset}_w_ckpt --gpu {args.gpu} --resume pretrained/cifar10_wide.pth.tar --out-dataset {ood_dataset} --wide", shell=True)

# DenseNet on CIFAR-100
for ood_dataset in ['Imagenet', 'Imagenet_resize', 'LSUN', 'LSUN_resize', 'iSUN']:
    print (f"CIFAR-100 {ood_dataset}")
    subprocess.run(f"python train.py -c checkpoints/cifar100_{ood_dataset}_w_ckpt --gpu {args.gpu} --resume pretrained/cifar100_wide.pth.tar --in-dataset cifar100 --out-dataset {ood_dataset} --wide", shell=True)