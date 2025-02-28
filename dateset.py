import torch
import numpy as np
from transformers import BertTokenizer
from config import BERT_PATH, labels_moon, category_label

# 从预训练模型路径加载BERT分词器
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)


# 定义第一个数据集类 MyDataset，用于将DataFrame格式的数据转化为可迭代的Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        # 根据传入DataFrame中的'label'列，将标签转换为对应的数字编码
        self.labels = [category_label[str(label)] for label in df['label']]
        # 对DataFrame中的'text'列每条文本进行分词，并统一填充到固定长度（64），
        # truncation=True 表示超长文本截断，return_tensors="pt" 表示返回PyTorch tensor
        self.texts = [tokenizer(comment,
                                padding='max_length',
                                max_length=64,
                                truncation=True,
                                return_tensors="pt")
                      for comment in df['text']]

    def classes(self):
        # 返回所有标签列表
        return self.labels

    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.labels)

    def get_batch_labels(self, idx):
        # 根据索引获取对应的标签（注意这里返回的是numpy数组形式）
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # 根据索引获取对应的文本数据（已经过tokenizer处理后的字典）
        return self.texts[idx]

    def __getitem__(self, idx):
        # 根据索引返回一个样本的数据，包括处理后的文本和对应的标签
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


# 定义第二个数据集类 SentimentDataset，主要用于情感分析任务
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_len=128):
        # 保存传入的数据
        self.data = data
        # 使用全局定义好的tokenizer
        self.tokenizer = tokenizer
        # 设置文本的最大长度
        self.max_len = max_len
        # 提取DataFrame中的文本数据
        self.texts = data['text'].values
        # 将标签转换为0和1（假设 'pos' 表示正面，其他表示负面）
        self.labels = data['label'].apply(lambda x: 1 if x == 'pos' else 0).values

    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引获取文本和标签
        text = self.texts[idx]
        label = self.labels[idx]
        # 使用BERT分词器对文本进行编码，添加特殊标记，统一填充和截断到max_len
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # 返回一个字典，包含input_ids、attention_mask以及标签（转换为torch.long类型）
        return {
            'input_ids': encoding['input_ids'].flatten(),        # 将input_ids展平为一维tensor
            'attention_mask': encoding['attention_mask'].flatten(),  # 将attention_mask展平为一维tensor
            'label': torch.tensor(label, dtype=torch.long)
        }
