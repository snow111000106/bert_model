import torch
import numpy as np
import pandas as pd
from creat_model import BertClassifier
import read_data
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import logging

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)


BERT_PATH = './model/bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
# labels = {
#     'neg': 0,
#     'pos': 1
# }
labels = {'login': 0,
          'creatAccount': 1,
          'initPassword': 2,
          'clearAccount': 3,
          'addSubAccount': 4,
          'clearCache': 5,
          'performanceTest': 6,
          'autoTest': 7,
          'runTask': 8,
          'unitTest': 9
          }


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[str(label)] for label in df['label']]
        self.texts = [tokenizer(comment,
                                padding='max_length',
                                max_length=64,
                                truncation=True,
                                return_tensors="pt")
                      for comment in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


if __name__ == '__main__':

    # 导入并划分数据集
    np.random.seed(112)
    # df = pd.read_csv('./data/test.csv')
    # df = read_data.read_data_txt()
    df = read_data.read_data_xls()
    # 数据集分割8:1:1
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.7 * len(df)), int(.8 * len(df))])
#
#     df_train.reset_index(drop=True, inplace=True)
#     df_val.reset_index(drop=True, inplace=True)
#     df_test.reset_index(drop=True, inplace=True)

    # #查看数据集信息
    # print(df.head())
    # print(df_train.iloc[17197])
    # print(df['rating'].unique())
    # print(df.info(verbose=True, show_counts=True))

    # 训练模型train model
    # from creat_model import BertClassifier
    # from train import train
    #
    # EPOCHS = 5
    # model = BertClassifier(BERT_PATH=BERT_PATH)
    # LR = 1e-6
    # train(model, df_train, df_val, LR, EPOCHS)

    # # 评估模型
    # from evaluate import evaluate
    # model = BertClassifier(BERT_PATH=BERT_PATH)
    # model.load_state_dict(torch.load('model/test_bert_category_model.pth'))
    # evaluate(model, df_test)

    # # 单个测试
    # model = BertClassifier(BERT_PATH=BERT_PATH)
    # model.load_state_dict(torch.load('./model/test_model.pth'))
    # data = '今天天气真好啊'
    # bert_input = tokenizer(data, padding='max_length', max_length=64, truncation=True, return_tensors="pt")
    # mask = bert_input['attention_mask'].to('cpu')
    # input_id = bert_input['input_ids'].squeeze(1).to('cpu')
    # out = model(input_id, mask)
    # moon = out.argmax(dim=1).item()
    # print(moon)
    # 两数之和
    # a = [2,7,11,15]
    # target = 9
    # num = len(a)
    # for i in range(num):
    #     for j in range(num-i-1):
    #         if a[i] + a[i+j+1] == target:
    #             print([i, i+j+1])
    # 罗马数字转整数
    # num_map = {
    #     'M': 1000,
    #     'D': 500,
    #     'C': 100,
    #     'L': 50,
    #     'X': 10,
    #     'V': 5,
    #     'I': 1
    # }
    # num = 123
    # re = []
    # for i, v in num_map.items():
    #     n = int(num/v)
    #     num = num - n*v
    #     for j in range(n):
    #         re.append(i)
    # result = ''.join(re)
    # print(result)
    # 最长公共前缀
    # sorts = ['floww', 'flower', 'flow', 'ffa']
    # # sorts = ['dog', 'car', 'racecar']
    # min = len(sorts[0])
    # for i in sorts:
    #     l = len(i)
    #     if l < min:
    #         min = l
    # ww = ''
    # for j in range(min+1):
    #     num = sorts[0][:j]
    #     count = 0
    #     for word in sorts:
    #         if word.startswith(num):
    #             count += 1
    #     if count >= len(sorts):
    #         ww = num
    # return ww











