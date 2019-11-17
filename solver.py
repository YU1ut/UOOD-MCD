import os
import shutil
import time
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms

import models
from utils import Bar, AverageMeter, accuracy, mkdir_p


class Solver(object):
    def __init__(self, args):
        torch.backends.cudnn.benchmark = True
        self.args = args
        self.best_prec1 = 0
        if not os.path.isdir(self.args.checkpoint):
            mkdir_p(self.args.checkpoint)

        self.set_dataloater()

        self.set_model()

        self.set_optimizer()

        self.set_criterion()

        title = 'Proposed'
        if self.args.resume:
            print('==> Resuming from checkpoint..')
            assert os.path.isfile(self.args.resume), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(self.args.resume)
            self.g.load_state_dict(checkpoint['state_dict_g'])
            self.c1.load_state_dict(checkpoint['state_dict_c1'])
            self.c2.load_state_dict(checkpoint['state_dict_c2'])
            self.opt_g.load_state_dict(checkpoint['opt_g'])
            self.opt_c1.load_state_dict(checkpoint['opt_c1'])
            self.opt_c2.load_state_dict(checkpoint['opt_c2'])
            print("=> loaded checkpoint '{}'"
                    .format(self.args.resume))

    def set_dataloater(self):
        raise NotImplementedError

    def set_model(self):
        self.g = models.DenseNet().cuda()
        self.c1 = models.Classifier(self.g.in_planes, self.args.num_classes).cuda()
        self.c2 = models.Classifier(self.g.in_planes, self.args.num_classes).cuda()
        

    def set_optimizer(self):
        self.opt_g = torch.optim.SGD(self.g.parameters(), self.args.lr,
                                        momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)

        self.opt_c1 = torch.optim.SGD(self.c1.parameters(), self.args.lr,
                                        momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)

        self.opt_c2 = torch.optim.SGD(self.c2.parameters(), self.args.lr,
                                        momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)

    def set_criterion(self):
        self.criterion_bce = nn.BCELoss()
        self.criterion_cel = nn.CrossEntropyLoss().cuda()

    def clear_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()

    def set_train(self):
        self.g.train()
        self.c1.train()
        self.c2.train()
    
    def set_eval(self):
        self.g.eval()
        self.c1.eval()
        self.c2.eval()

    def discrepancy(self, out1, out2):
        probs1 = F.softmax(out1, dim=1)
        probs2 = F.softmax(out2, dim=1)
        L1 = -torch.mean(torch.sum(F.log_softmax(out1, dim=1) * probs1, dim=1))
        L2 = -torch.mean(torch.sum(F.log_softmax(out2, dim=1) * probs2, dim=1))
        prob_diff = torch.clamp(self.args.th - (L1 - L2), min=0)
        return prob_diff

    def train(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        self.set_train()

        semi_train_iter = iter(self.semi_train_loader)
        end = time.time()
        bar = Bar('Training', max=len(self.train_loader))
        for batch_idx, (data_s, target_s) in enumerate(self.train_loader):

            try:
                data_t, _  = semi_train_iter.next()
            except:
                semi_train_iter = iter(self.semi_train_loader)
                data_t, _ = semi_train_iter.next()

            
            data_time.update(time.time() - end)

            data_s, target_s = data_s.cuda(), target_s.cuda(non_blocking=True)
            data_t = data_t.cuda()

            batch_size_s = len(target_s)
            
            # Step A
            feat_s = self.g(data_s)
            output_s = self.c1(feat_s)
            output_s2 = self.c2(feat_s.detach())

            loss_cel = self.criterion_cel(output_s, target_s)

            loss_cel2 = self.criterion_cel(output_s2, target_s)

            loss =  loss_cel + loss_cel2
            
            self.clear_grad()
            loss.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()

            # Step B
            
            feat_s = self.g(data_s)
            output_s = self.c1(feat_s)
            output_s2 = self.c2(feat_s)
            

            feat_t = self.g(data_t)
            output_t = self.c1(feat_t)
            output_t2 = self.c2(feat_t)

            loss_cel = self.criterion_cel(output_s, target_s)

            loss_cel2 = self.criterion_cel(output_s2, target_s)

            loss_dis = self.discrepancy(output_t, output_t2)
            loss = loss_cel + loss_cel2 + loss_dis

            self.clear_grad()
            loss.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            
            losses.update(loss_dis.item(), batch_size_s)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
                        batch=batch_idx + 1,
                        size=len(self.train_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        )
            bar.next()

        bar.finish()
        return losses.avg

    def val(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.set_eval()

        gts = []
        probs = []
        probs2 = []

        end = time.time()
        bar = Bar('Testing ', max=len(self.val_loader))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                # measure data loading time
                data_time.update(time.time() - end)
                data, target = data.cuda(), target.cuda(non_blocking=True)
                feat_s = self.g(data)
                output = self.c1(feat_s)
                output2 = self.c2(feat_s)

                prob = F.softmax(output, dim=1)
                prob2 = F.softmax(output2, dim=1)

                for i in range(len(output)):
                    gts.append(target[i].item())
                    probs.append(prob[i].cpu().numpy())
                    probs2.append(prob2[i].cpu().numpy())
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
            
                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                            batch=batch_idx + 1,
                            size=len(self.val_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            )
                bar.next()
            bar.finish()

        data = {'gts':gts, 'probs':probs, 'probs2':probs2}

        #calculate the L1 distance
        diff = [0]
        cifar = []
        other = []
        for i in range(len(data['gts'])):
            gt = data['gts'][i]
            probs = data['probs'][i]
            ema_probs = data['probs2'][i]
            if gt >= 0 and gt < self.args.num_classes:
                cifar.append(sum(np.abs(probs-ema_probs)))
            else:
                other.append(sum(np.abs(probs-ema_probs)))
            diff.append(sum(np.abs(probs-ema_probs))+10e-5)
        diff = sorted(list(set(diff)))[::-1]
        cifar, other = np.array(cifar), np.array(other)

        #calculate the AUROC
        aurocBase = 0.0
        fprTemp = 1.0
        for delta in diff:
            tpr = np.sum(np.sum(cifar < delta)) / np.float(len(cifar))
            fpr = np.sum(np.sum(other < delta)) / np.float(len(other))
            aurocBase += (-fpr+fprTemp)*tpr
            fprTemp = fpr
        aurocBase += fpr * tpr

        print (f"Val AUROC: {aurocBase} ")
        prec = aurocBase
        is_best = prec > self.best_prec1
        if is_best:
            self.is_best = True
        else:
            self.is_best = False
        self.best_prec1 = max(prec, self.best_prec1)
        return 

    def test(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.set_eval()

        gts = []
        probs = []
        probs2 = []

        end = time.time()
        bar = Bar('Testing ', max=len(self.test_loader))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                # measure data loading time
                data_time.update(time.time() - end)
                data, target = data.cuda(), target.cuda(non_blocking=True)
                feat_s = self.g(data)
                output = self.c1(feat_s)
                output2 = self.c2(feat_s)

                prob = F.softmax(output, dim=1)
                prob2 = F.softmax(output2, dim=1)

                for i in range(len(output)):
                    gts.append(target[i].item())
                    probs.append(prob[i].cpu().numpy())
                    probs2.append(prob2[i].cpu().numpy())
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
            
                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                            batch=batch_idx + 1,
                            size=len(self.test_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            )
                bar.next()
            bar.finish()

        self.save_checkpoint({
            'epoch': epoch,
            'state_dict_g': self.g.state_dict(),
            'state_dict_c1': self.c1.state_dict(),
            'state_dict_c2': self.c2.state_dict(),
            'best_prec1': self.best_prec1,
            'opt_g' : self.opt_g.state_dict(),
            'opt_c1' : self.opt_c1.state_dict(),
            'opt_c2' : self.opt_c2.state_dict(),
        }, self.is_best, checkpoint=self.args.checkpoint)

        dic = {'gts':gts, 'probs':probs,  'probs2':probs2,}
        pickle.dump(dic, open( os.path.join(self.args.checkpoint,"results.p"), "wb" ) )

        return 

    def save_checkpoint(self, state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

    def adjust_learning_rate(self, epoch):
        for param_group in self.opt_g.param_groups:
            param_group['lr'] = self.args.lr
        for param_group in self.opt_c1.param_groups:
            param_group['lr'] = self.args.lr
        for param_group in self.opt_c2.param_groups:
            param_group['lr'] = self.args.lr