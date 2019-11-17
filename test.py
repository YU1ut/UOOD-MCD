import numpy as np
import time
from scipy import misc
import pickle
import torch

import click

diff = [-0.1, 0, 2.1]

@click.command()
@click.option('--result', type=str, required=True)
def main(result):
    global diff
    data = pickle.load(open(f"{result}/results.p", "rb" ))
    cifar = []
    cifar_d = []
    other = []
    other_d = []
    for i in range(len(data['gts'])):
        gt = data['gts'][i]
        probs = data['probs'][i]
        ema_probs = data['probs2'][i]
        if gt >= 0:
            cifar.append(sum(np.abs(probs-ema_probs)))
            cifar_d.append(-np.sum(np.log(probs) * probs)+np.sum(np.log(ema_probs) * ema_probs))
        else:
            other.append(sum(np.abs(probs-ema_probs)))
            other_d.append(-np.sum(np.log(probs) * probs)+np.sum(np.log(ema_probs) * ema_probs))
        diff.append(sum(np.abs(probs-ema_probs))+10e-5)
    diff = sorted(list(set(diff)))[::-1]
    cifar, other = np.array(cifar), np.array(other)
    cifar_d, other_d = np.array(cifar_d), np.array(other_d)
    print (f"#All: {len(data['gts'])} #Cifar: {len(cifar)} #Other: {len(other)}")
    print (f"IN_D:{np.mean(cifar_d)}, OOD_D:{np.mean(other_d)}")

    fpr = tpr95(cifar, other)
    error = detection(cifar, other)
    auroc_ = auroc(cifar, other)
    auprin = auprIn(cifar, other)
    auprout = auprOut(cifar, other)
     
    print("{:20}{:13.1f} ".format("FPR at TPR 95%:",fpr*100))
    print("{:20}{:13.1f}".format("Detection error:",error*100))
    print("{:20}{:13.1f}".format("AUROC:",auroc_*100))
    print("{:20}{:13.1f}".format("AUPR In:",auprin*100))
    print("{:20}{:13.1f}".format("AUPR Out:",auprout*100))

def tpr95(X1, Y1):
    #calculate the falsepositive error when tpr is 95%
    total = 0.0
    fpr = 0.0
    for delta in diff:
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 <= delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
    fprBase = fpr/total

    return fprBase

def auroc(X1, Y1):
    #calculate the AUROC
    aurocBase = 0.0
    fprTemp = 1.0
    for delta in diff:
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        aurocBase += (-fpr+fprTemp)*tpr
        fprTemp = fpr
    aurocBase += fpr * tpr

    return aurocBase

def auprIn(X1, Y1):
    #calculate the AUPR
    precisionVec = []
    recallVec = []
    auprBase = 0.0
    recallTemp = 1.0
    for delta in diff:
        tp = np.sum(np.sum(X1 <= delta))
        fp = np.sum(np.sum(Y1 <= delta))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp / len(X1)
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase

def auprOut(X1, Y1):
    #calculate the AUPR
    auprBase = 0.0
    recallTemp = 1.0
    for delta in diff[::-1]:
        fp = np.sum(np.sum(X1 > delta))
        tp = np.sum(np.sum(Y1 > delta))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp / len(Y1)
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision
        
    return auprBase


def detection(X1, Y1):
    #calculate the minimum detection error
    errorBase = 1.0
    for delta in diff:
        tpr_ = np.sum(np.sum(X1 > delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr_+error2)/2.0)

    return errorBase

if __name__ == '__main__':
    main()
