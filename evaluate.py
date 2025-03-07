# -*- coding: utf-8 -*-
# @Time    : 2023/5/19
# @Author  : 陈雪虹
# @File    : evaluate.py

import torch
from dateset import MyDataset, SentimentDataset, ForVecDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


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


def evaluate_moon(model, test_data):
    """
    对模型在测试集上的表现进行评估，计算准确率、混淆矩阵、精确率、召回率和F1分数。
    这里使用 SentimentDataset 以保持与训练过程的一致性。
    """
    # 使用 SentimentDataset 构造测试数据集
    test_dataset = SentimentDataset(test_data)
    # 创建 DataLoader，batch_size 设置为2
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2)

    # 判断是否有可用的GPU，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_acc_test = 0  # 用于累计预测正确的样本数量
    y_true = []  # 存储所有真实标签
    y_scores = []  # 存储预测为正类的概率（用于后续评价指标计算）
    y_preds = []  # 存储模型的预测类别

    # 在不计算梯度的环境下进行评估，加快速度并节省内存
    with torch.no_grad():
        # 使用 tqdm 显示评估进度
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # 从batch中获取input_ids、attention_mask和标签，并移动到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 模型前向传播，获得输出结果
            output = model(input_ids, attention_mask)

            # 计算当前batch预测正确的样本数，并累加到总数中
            correct = (output.argmax(dim=1) == labels).sum().item()
            total_acc_test += correct

            # 使用 softmax 将输出转换为概率，并获取正类的概率（假设正类在索引1处）
            probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            # 将概率、真实标签和预测结果加入到对应列表中
            y_scores.extend(probs.tolist())
            y_true.extend(labels.cpu().numpy().tolist())
            y_preds.extend(output.argmax(dim=1).cpu().numpy().tolist())

    # 计算测试集的总体准确率（正确预测的样本数 / 测试集样本总数）
    test_accuracy = total_acc_test / len(test_dataset)
    print(f'Test Accuracy: {test_accuracy:.3f}')

    # 计算混淆矩阵、精确率、召回率和F1分数
    cm = confusion_matrix(y_true, y_preds)
    precision = precision_score(y_true, y_preds, average='macro')
    recall = recall_score(y_true, y_preds, average='macro')
    f1 = f1_score(y_true, y_preds, average='macro')

    print(f'Confusion Matrix:\n{cm}')
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')


def evaluate_moon_for_vec(model, test_data):
    test_dataset = ForVecDataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_acc_test = 0
    y_true, y_scores, y_preds = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            text_vectors = batch['input_vectors'].to(device)
            labels = batch['label'].to(device)

            output = model(text_vectors)  # 直接输入 Word2Vec 向量

            correct = (output.argmax(dim=1) == labels).sum().item()
            total_acc_test += correct

            probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            y_scores.extend(probs.tolist())
            y_true.extend(labels.cpu().numpy().tolist())
            y_preds.extend(output.argmax(dim=1).cpu().numpy().tolist())

    test_accuracy = total_acc_test / len(test_dataset)
    print(f'Test Accuracy: {test_accuracy:.3f}')

    cm = confusion_matrix(y_true, y_preds)
    precision = precision_score(y_true, y_preds, average='macro')
    recall = recall_score(y_true, y_preds, average='macro')
    f1 = f1_score(y_true, y_preds, average='macro')

    print(f'Confusion Matrix:\n{cm}')
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')




