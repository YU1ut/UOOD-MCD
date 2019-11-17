from __future__ import print_function
import argparse
import os
import torch
from torchvision import transforms
from torchvision import datasets
from solver import Solver
import models
from utils import CIFAR10Mix, CIFAR100Mix
import argparse
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='number of known classes (default: 90)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 90)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--th', type=float, default=1.2, metavar='TH',
                    help='threshold of discrepancy (default: 1.2)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-c', '--checkpoint', default='cifar10_Imagenet_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: pretrain_checkpoint)')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--wide', dest='wide', action='store_true',
                    help='use wide-resnet')
parser.add_argument('--in-dataset', default="cifar10", type=str,
                    help='training set')
parser.add_argument('--out-dataset', default="Imagenet", type=str,
                    help='out-of-distribution dataset')

args = parser.parse_args()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()


class MySolver(Solver):
    def __init__(self, args):
        super(MySolver, self).__init__(args)

    
    def set_model(self):
        if args.wide:
            self.g = models.WideResNet().cuda()
            self.c1 = models.Classifier(self.g.nChannels, self.args.num_classes).cuda()
            self.c2 = models.Classifier(self.g.nChannels, self.args.num_classes).cuda()
        else:
            self.g = models.DenseNet().cuda()
            self.c1 = models.Classifier(self.g.in_planes, self.args.num_classes).cuda()
            self.c2 = models.Classifier(self.g.in_planes, self.args.num_classes).cuda()

    def set_dataloater(self):
        if args.in_dataset == 'cifar10': 
            train_dataset = datasets.CIFAR10('data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                    ]))

            semi_train_dataset = CIFAR10Mix('data', f"./data/{args.out_dataset}", train=False, val=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                            ]))

            test_dataset = CIFAR10Mix('data', f"./data/{args.out_dataset}", train=False, val=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                            ]))

            val_dataset = CIFAR10Mix('data', f"./data/{args.out_dataset}", train=False, val=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                            ]))

        else:
            args.num_classes = 100
            train_dataset = datasets.CIFAR100('data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                    ]))

            semi_train_dataset = CIFAR100Mix('data', f"./data/{args.out_dataset}", train=False, val=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                            ]))

            test_dataset = CIFAR100Mix('data', f"./data/{args.out_dataset}", train=False, val=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                            ]))

            val_dataset = CIFAR100Mix('data', f"./data/{args.out_dataset}", train=False, val=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
                            ]))
        

        self.train_loader = torch.utils.data.DataLoader(train_dataset, 
            batch_size=args.batch_size, shuffle=True, num_workers=0)

        self.semi_train_loader = torch.utils.data.DataLoader(semi_train_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=0)

        self.test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=args.batch_size, shuffle=False, num_workers=0)

        self.val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=args.batch_size, shuffle=False, num_workers=0)

if __name__ == '__main__':
    solver = MySolver(args)
    for epoch in range(args.start_epoch, args.epochs):
        solver.adjust_learning_rate(epoch)
        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))
        train_loss = solver.train(epoch)
        solver.val(epoch)
        if solver.is_best:
            solver.test(epoch)

    print('Best val acc:')
    print(solver.best_prec1)