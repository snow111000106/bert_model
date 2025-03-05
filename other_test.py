import pandas as pd
import random
import torch
from transformers import BertModel, BertTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# 加载 BERT-base-Chinese 模型和分词器
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 定义两句话
sentence1 = "小米粥很好吃"
sentence2 = "小米配馒头不好吃"

# 分词并获取 token 索引
inputs1 = tokenizer(sentence1, return_tensors="pt")
inputs2 = tokenizer(sentence2, return_tensors="pt")

# 计算 BERT 词向量
with torch.no_grad():
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

# 获取 tokenized 结果
tokens1 = tokenizer.convert_ids_to_tokens(inputs1["input_ids"][0])
tokens2 = tokenizer.convert_ids_to_tokens(inputs2["input_ids"][0])

# 打印分词结果
print(f"句子1 Tokenized: {tokens1}")
print(f"句子2 Tokenized: {tokens2}")

# 找到 '苹' 和 '果' 的索引
idxs1 = [i for i, token in enumerate(tokens1) if token in ["小", "米"]]
idxs2 = [i for i, token in enumerate(tokens2) if token in ["小", "米"]]

# 提取'苹果'的词向量（如果有多个 token，则取均值）
vector1 = torch.mean(outputs1.last_hidden_state[0, idxs1, :], dim=0)
vector2 = torch.mean(outputs2.last_hidden_state[0, idxs2, :], dim=0)

# 转换为 numpy 数组
vector1 = vector1.numpy().reshape(1, -1)
vector2 = vector2.numpy().reshape(1, -1)

# 计算余弦相似度
similarity = cosine_similarity(vector1, vector2)[0][0]

# 输出结果
print(f"余弦相似度: {similarity}")



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