import pandas as pd
import random
from transformers import BertModel, BertTokenizer
import numpy as np


def read_data_txt():
    data = []
    with open('./data/tsinghua.negative.gb.txt', 'r', encoding='gbk') as nf:
        for line in nf:
            data.append([line.strip(), 'neg'])

    with open('./data/tsinghua.positive.gb.txt', 'r', encoding='gbk') as pf:
        for line in pf:
            data.append([line.strip(), 'pos'])
    random.shuffle(data)
    col = ['text', 'label']
    data = pd.DataFrame(data)
    data.columns = col
    return data


def read_data_xls():
    df = pd.read_excel('./data/category_train.xls')
    data = df.apply(lambda x: x.astype(str).str.replace(r'\d+', '', regex=True))
    return data

# path = './model/bert-base-chinese'
# tokenizer = BertTokenizer.from_pretrained(path)
#
# # bert = BertModel.from_pretrained(path)
#
# test = '我爱你我的中国呀'
# print(tokenizer.tokenize(test))
# bert_input = tokenizer(test, padding='max_length', max_length=10, truncation=True, return_tensors="pt")
# print(bert_input['input_ids'])
# print(bert_input['token_type_ids'])
# print(bert_input['attention_mask'])

# np.random.seed(112)
# df = pd.read_csv('./data/train.csv')[:10]
# df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
#                                      [int(.8 * len(df)), int(.9 * len(df))])
#
# df_train.reset_index(drop=True, inplace=True)
# df_val.reset_index(drop=True, inplace=True)
# df_test.reset_index(drop=True, inplace=True)
# print(df_train)
# a = read_data2()
# print(a)