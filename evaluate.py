# -*- coding: utf-8 -*-
# @Time    : 2023/5/19
# @Author  : 陈雪虹
# @File    : evaluate.py

import torch
from dateset import MyDataset, SentimentDataset
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def evaluate(model, test_data):
    # 评估模型

    test = MyDataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    total_acc_test = 0
    y_true = []
    y_scores = []
    y_preds = []
    with torch.no_grad():
        for test_input, test_label in tqdm(test_dataloader):

            test_label = test_label.to(device)
            mask = test_input['attention_mask'].squeeze(1).to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            # 计算准确率
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
            # 获取预测概率
            probs = torch.softmax(output, dim=1)[:, 1].numpy()  # 获取正类的概率
            y_scores.extend(probs)
            y_true.extend(test_label.numpy())
            y_preds.extend(output.argmax(dim=1).numpy())
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    # 计算混淆矩阵、精确率、召回率和F1分数
    cm = confusion_matrix(y_true, y_preds)
    precision = precision_score(y_true, y_preds, average='macro')
    recall = recall_score(y_true, y_preds, average='macro')
    f1 = f1_score(y_true, y_preds, average='macro')

    print(f'Confusion Matrix:\n{cm}')
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')

    # # 绘制混淆矩阵
    # label_names = list(category_label.keys())
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()

    # fpr, tpr, _ = roc_curve(y_true, y_scores)
    # roc_auc = auc(fpr, tpr)
    # # 计算混淆矩阵、精确率、召回率和F1分数
    # cm = confusion_matrix(y_true, y_preds)
    # precision = precision_score(y_true, y_preds)
    # recall = recall_score(y_true, y_preds)
    # f1 = f1_score(y_true, y_preds)
    #
    # print(f'Confusion Matrix:\n{cm}')
    # print(f'Precision: {precision:.3f}')
    # print(f'Recall: {recall:.3f}')
    # print(f'F1 Score: {f1:.3f}')
    # # 二分类绘制混淆矩阵
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()

    # # 二分类绘制ROC曲线
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()


def evaluate_moon(model,  test_data):

    test = SentimentDataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()

    all_preds = []
    all_labels = []

    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(input_ids)

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {accuracy:.4f}')



