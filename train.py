# -*- coding: utf-8 -*-
# @Time    : 2023/5/19
# @Author  : 陈雪虹
# @File    : train.py

import torch
from torch.optim import Adam
from torch import nn
from tqdm import tqdm
from transformers import AdamW
from dateset import MyDataset,SentimentDataset


def train(model, train_data, val_data, learning_rate, epochs):
    # 通过Dataset类获取训练和验证集
    train, val = MyDataset(train_data), MyDataset(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器i
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    best_val_acc = 0

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    # 开始进入训练循环
    for epoch_num in range(epochs):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0

        # 进度条函数tqdm
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].squeeze(1).to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            # 通过模型得到输出
            output = model(input_id, mask)
            # 计算损失
            train_label = train_label.to(torch.int64)
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # 模型更新
            model.zero_grad()

            batch_loss.backward()
            optimizer.step()
            # torch.cuda.empty_cache()

        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失
        total_acc_val = 0
        total_loss_val = 0
        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for val_input, val_label in val_dataloader:
                # 如果有GPU，则使用GPU，接下来的操作同训练
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].squeeze(1).to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                val_label = val_label.to(torch.int64)
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        if total_acc_val > best_val_acc:
            best_val_acc = total_acc_val
            torch.save(model.state_dict(), 'model/test_bert_category_model.pth')

        print(
            f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')


def train_moon(model, train_data, val_data, lr, epochs):
    # 使用 SentimentDataset 类将训练数据和验证数据转为数据集对象
    train_dataset = SentimentDataset(train_data)
    val_dataset = SentimentDataset(val_data)

    # 创建 DataLoader，用于按批次加载数据，训练时设置 shuffle=True 打乱数据
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2)

    # 判断是否有可用的GPU，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 将模型加载到对应设备上
    model = model.to(device)

    # 定义优化器，这里使用AdamW，并设置学习率lr
    optimizer = AdamW(model.parameters(), lr=lr)
    # 定义交叉熵损失函数，适用于分类任务
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = 0  # 用于记录验证集上最好的准确率

    # 开始训练epochs轮
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        total_loss_train = 0  # 累计训练损失
        total_acc_train = 0   # 累计训练正确预测的样本数

        # 使用 tqdm 显示当前训练轮次的进度条
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()  # 清空梯度

            # 获取batch中的输入数据，并将其移动到设备上
            input_ids = batch['input_ids'].to(device)
            # 如果input_ids多了一个维度，则压缩掉（例如：[1, 1, seq_len] -> [1, seq_len]）
            if input_ids.dim() == 3:
                input_ids = input_ids.squeeze(1)
            attention_mask = batch['attention_mask'].to(device)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(1)
            labels = batch['label'].to(device)

            # 前向传播：传入input_ids和attention_mask，获得模型输出
            outputs = model(input_ids, attention_mask)
            # 计算当前batch的损失
            loss = criterion(outputs, labels)
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 根据梯度更新模型参数

            # 累计当前batch的损失和正确预测的样本数
            total_loss_train += loss.item()
            total_acc_train += (outputs.argmax(dim=1) == labels).sum().item()

        # 计算训练集的平均损失和准确率
        train_loss = total_loss_train / len(train_dataset)
        train_acc = total_acc_train / len(train_dataset)

        # 开始验证阶段，设置模型为评估模式，不启用梯度计算
        model.eval()
        total_loss_val = 0  # 累计验证损失
        total_acc_val = 0   # 累计验证正确预测的样本数
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                if input_ids.dim() == 3:
                    input_ids = input_ids.squeeze(1)
                attention_mask = batch['attention_mask'].to(device)
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask.squeeze(1)
                labels = batch['label'].to(device)

                # 前向传播获取验证结果
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_loss_val += loss.item()
                total_acc_val += (outputs.argmax(dim=1) == labels).sum().item()

        # 计算验证集的平均损失和准确率
        val_loss = total_loss_val / len(val_dataset)
        val_acc = total_acc_val / len(val_dataset)

        # 如果本轮验证准确率更高，则保存当前模型参数
        if total_acc_val > best_val_acc:
            best_val_acc = total_acc_val
            torch.save(model.state_dict(), 'model/test_bert_cnn_moon_model.pth')

        # 打印当前轮次的训练和验证的损失及准确率
        print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}")


def train_moon_2(model, train_data, lr, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_loader = torch.utils.data.DataLoader(SentimentDataset(train_data), batch_size=2, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = (batch['input_ids'].squeeze(1).to(device),batch['attention_mask'].squeeze(1).to(device), batch['label'].to(device))
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
        train_acc = train_correct / len(train_data)
        torch.save(model.state_dict(), 'model/test_bert_cnn_moon_model_2.pth')
        print(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.3f} | Train Acc {train_acc:.3f}")

