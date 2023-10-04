import torch
import math
import argparse
from icecream import ic
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt   

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from modules.deepromoter import DeePromoter
from dataloader import load_data, load_data_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(net, loaders):
    """
    Infer and check results against labels
    :param net: Model object in eval state
    :param loaders: List of torch dataloader for infer
    :return: List of [correct, total] for every dataloader, list of predicted results in int type
    """
    eval_result = list()
    ltotal = list()
    lcorrect = list()
    pred_result = list()
    for load in loaders:
        total = 0
        correct = 0
        pred_list = list()
        for data in load:
            inputs = data[0]
            labels = data[1]
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_list += list(predicted.cpu().numpy())
        acc = correct/total
        eval_result.append(acc)
        lcorrect.append(correct)
        ltotal.append(total)
        pred_result.append(pred_list)
    return (lcorrect, ltotal), pred_result


def mcc(data):
    """
    Calculate Matthew correlation coeficient
    data: List output of evaluate with the first item is positive result and second item is negative result
    return: Precision, recall, MCC
    """
    pos_count = data[0][0]
    neg_count = data[0][1]

    tol_pos_count = data[1][0]
    tol_neg_count = data[1][1]

    TP = pos_count
    FN = tol_pos_count - pos_count
    TN = neg_count
    FP = tol_neg_count - neg_count
    #ic(TP)
    #ic(FP)
    #ic(FN)
    #ic(TN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return precision, recall, MCC


def test(data_path, pretrain, ker=None):
    if ker is None:
        ker = [16, 8, 4]

    dataloader = load_data_test(data_path, device=device)

    # model define
    net = DeePromoter(ker, drop = 0.4)
    net.to(device)

    net.load_state_dict(torch.load(pretrain))

    net.eval()
    eval_data, results = evaluate(net, [dataloader])

    return eval_data, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="path to dataset(txt file)",
    )
    parser.add_argument("-w", "--weight", type=str, help="Path to pre-train")
    args = parser.parse_args()

    data = load_data(args.data, device=device)
    train_pos, val_pos, test_pos, train_neg, val_neg, test_neg = data
    # model define
    ker = [16, 8, 4]
    net = DeePromoter(ker)
    net.to(device)
    net.load_state_dict(torch.load(args.weight, map_location=torch.device('cpu')))
    net.eval()
    train_set = [train_pos, train_neg]
    val_set = [val_pos, val_neg]
    test_set = [test_pos, test_neg]
  
    eval_data, results = evaluate(net, test_set)
    precision, recall, MCC = mcc(eval_data)

    with open("infer_test_results.txt", "w") as f:
        print("Save the results below to infer_test_results.txt\n")
        f.write('Precision: '+ str(precision) + "\n")
        f.write('Recall: '+ str(recall) + "\n")
        f.write('MCC: '+ str(MCC) + "\n")
        
        print('Test Precision: '+ str(precision) + "\n")
        print('Test Recall: '+ str(recall) + "\n")
        print('Test MCC: '+ str(MCC) + "\n")

        for out in eval_data:
            f.write('Number of [correct predictions, total]:' + str(out) + "\n")
           # print('Number of [correct predictions, total]:' + str(out) + "\n")

        for out in results:
          f.write(str(out))
          f.write("\n")
          # print(out, '\n')

def plot_mat(data_path, pretrain):
    # model define
    ker = [16,8,4]
    net = DeePromoter(ker, drop=0.5)
    net.to(device)
    net.load_state_dict(torch.load(pretrain))
    net.eval()
    data = load_data(data_path, batch_size=32, device=device)
    train_pos, val_pos, test_pos, train_neg, val_neg, test_neg = data
    test_set = [test_pos, test_neg]
    val_set = [val_pos,val_neg]
    eval_data, results = evaluate(net, test_set)

    # y_true = np.append(np.ones(294), np.zeros(294)) # human TATA
    y_true = np.append(np.ones(2593), np.zeros(2593)) # human non TATA
    y_pred = np.append(results[0],results[1])

    print('\nTest Classification Report:\n\n', classification_report(y_true, y_pred, digits=4))

    fig, ax = plt.subplots(figsize=(8,5))
    sn.heatmap(confusion_matrix(y_true, y_pred), annot = True, fmt = 'g', ax=ax)
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Non-Promoter', 'Promoter']); ax.yaxis.set_ticklabels(['Non-Promoter', 'Promoter']);
    plt.show()

    
   

        
