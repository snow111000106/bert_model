from torch import nn
from transformers import BertModel
import torch


class BertClassifier(nn.Module):
    def __init__(self, bert_path, dropout=0.3):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 10)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        # print(input_id.shape)  # 应该是 [batch_size, sequence_length]
        # print(mask.shape)  # 应该是 [batch_size, sequence_length]

        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


class CNN_BERT_Model(nn.Module):
    def __init__(self, bert_path, dropout=0.5):
        super(CNN_BERT_Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask):
        # BERT 输出 shape: (batch_size, seq_len, hidden_size)
        x = self.bert(input_ids, attention_mask=attention_mask)[0]
        # 转换为 (batch_size, hidden_size, seq_len)
        x = x.permute(0, 2, 1)
        # 卷积操作，输出 shape: (batch_size, 256, L) (L = seq_len - kernel_size + 1)
        x = self.conv1(x)
        # 全局最大池化，沿着序列长度维度取最大值，输出 shape: (batch_size, 256)
        x = torch.max(x, dim=2)[0]
        x = self.dropout(x)
        x = self.fc(x)
        return x

