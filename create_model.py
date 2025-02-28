from torch import nn
from transformers import BertModel
import torch


class BertClassifier(nn.Module):
    def __init__(self, bert_path, dropout=0.3):
        super(BertClassifier, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(bert_path)
        # 定义Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 定义全连接层，将BERT的768维输出映射到10个类别（例如10分类任务）
        self.linear = nn.Linear(768, 10)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        # 通过BERT模型获取输出：
        # 返回的第一个元素为每个token的特征表示，
        # 第二个元素（pooled_output）为整体句子的表示（通常是[CLS]对应的向量）
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        # 对池化后的输出应用Dropout
        dropout_output = self.dropout(pooled_output)
        # 通过全连接层得到线性输出
        linear_output = self.linear(dropout_output)
        # 应用ReLU激活函数获得最终输出
        final_layer = self.relu(linear_output)
        return final_layer


class CNN_BERT_Model(nn.Module):
    def __init__(self, bert_path, dropout=0.5):
        super(CNN_BERT_Model, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(bert_path)
        # 定义1维卷积层：
        # 输入通道数为768（BERT的隐藏层维度），输出通道数为256，卷积核大小为3
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3)
        # 定义Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 定义全连接层，将卷积后经过池化的特征映射到2个类别（二分类任务）
        self.fc = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask):
        # 通过BERT模型获取最后一层隐藏状态，输出形状为 [batch_size, seq_length, hidden_size]
        x = self.bert(input_ids, attention_mask=attention_mask)[0]
        # 调整维度顺序，转换为 [batch_size, hidden_size, seq_length]，以便后续进行1D卷积
        x = x.permute(0, 2, 1)
        # 通过1维卷积层提取局部特征，输出形状为 [batch_size, 256, L] (L取决于序列长度和卷积核大小)
        x = self.conv1(x)
        # 采用全局最大池化，在序列维度上取最大值，得到形状 [batch_size, 256]
        x = torch.max(x, dim=2)[0]
        # 对池化后的特征应用Dropout
        x = self.dropout(x)
        # 通过全连接层得到最终的分类输出
        x = self.fc(x)
        return x
