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
    train_dataset = SentimentDataset(train_data)
    val_dataset = SentimentDataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss_train = 0
        total_acc_train = 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            if input_ids.dim() == 3:
                input_ids = input_ids.squeeze(1)
            attention_mask = batch['attention_mask'].to(device)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(1)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            total_acc_train += (outputs.argmax(dim=1) == labels).sum().item()

        train_loss = total_loss_train / len(train_dataset)
        train_acc = total_acc_train / len(train_dataset)

        model.eval()
        total_loss_val = 0
        total_acc_val = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                if input_ids.dim() == 3:
                    input_ids = input_ids.squeeze(1)
                attention_mask = batch['attention_mask'].to(device)
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask.squeeze(1)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_loss_val += loss.item()
                total_acc_val += (outputs.argmax(dim=1) == labels).sum().item()

        val_loss = total_loss_val / len(val_dataset)
        val_acc = total_acc_val / len(val_dataset)

        if total_acc_val > best_val_acc:
            best_val_acc = total_acc_val
            torch.save(model.state_dict(), 'model/test_bert_cnn_moon_model.pth')

        print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}")