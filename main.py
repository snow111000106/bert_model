import torch
import numpy as np
import pandas as pd
from config import BERT_PATH
import read_data
from transformers import BertTokenizer
import logging
from sklearn.model_selection import train_test_split
import dateset
from torch.utils.data import DataLoader


# loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
# for logger in loggers:
#     if "transformers" in logger.name.lower():
#         logger.setLevel(logging.ERROR)


def run_category_train(types):
    from create_model import BertClassifier
    from train import train
    from evaluate import evaluate

    np.random.seed(112)
    df = read_data.read_data_xls()
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.7 * len(df)), int(.8 * len(df))])
    # 打散
    # df_train.reset_index(drop=True, inplace=True)
    # df_val.reset_index(drop=True, inplace=True)
    # df_test.reset_index(drop=True, inplace=True)

    # 查看数据集信息
    # print(df.head())
    # print(df_train.iloc[17197])
    # print(df['rating'].unique())
    # print(df.info(verbose=True, show_counts=True))
    model = BertClassifier(bert_path=BERT_PATH)

    if types == 'train':
        EPOCHS = 5
        LR = 1e-6
        train(model, df_train, df_val, LR, EPOCHS)

    elif types == 'evaluate':
        model.load_state_dict(torch.load('model/test_bert_category_model.pth'))
        evaluate(model, df_test)

    elif types == 'test':
        # 单个测试
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        model = BertClassifier(bert_path=BERT_PATH)
        model.load_state_dict(torch.load('./model/test_bert_category_model.pth'))
        data = '清掉缓存数据'
        bert_input = tokenizer(data, padding='max_length', max_length=64, truncation=True, return_tensors="pt")
        mask = bert_input['attention_mask'].to('cpu')
        input_id = bert_input['input_ids'].squeeze(1).to('cpu')
        out = model(input_id, mask)
        re = out.argmax(dim=1).item()
        print(re)


def run_mon_train(types):
    from create_model import CNN_BERT_Model
    from train import train_moon
    from evaluate import evaluate_moon

    data = read_data.read_data_txt()
    df_train, df_val, df_test = np.split(data.sample(frac=1, random_state=42),
                                         [int(.7 * len(data)), int(.8 * len(data))])

    model = CNN_BERT_Model(bert_path=BERT_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if types == 'train':
        LR = 1e-6
        train_moon(model, df_train, LR, 2)

    elif types == 'evaluate':
        model.load_state_dict(torch.load('./model/test_bert_cnn_moon_model.pth'))
        evaluate_moon(model, df_test)

    elif types == 'test':
        # 单个测试

        tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        model = CNN_BERT_Model(bert_path=BERT_PATH)
        model.load_state_dict(torch.load('./model/test_bert_cnn_moon_model.pth'))
        data = '这个水果手机的系统很流畅。'
        bert_input = tokenizer(data, padding='max_length', max_length=64, truncation=True, return_tensors="pt")
        mask = bert_input['attention_mask'].to('cpu')
        input_id = bert_input['input_ids'].squeeze(1).to('cpu')

        out = model(input_id, mask)
        moon = out.argmax(dim=1).item()
        print(moon)

        # from transformers import BertModel
        # 词向量查看
        # tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        # model = BertModel.from_pretrained(BERT_PATH)
        #
        # # 获取词嵌入矩阵
        # embeddings = model.embeddings.word_embeddings.weight  # 形状通常为 [vocab_size, hidden_size]
        # print("词嵌入矩阵的形状：", embeddings.shape)
        #
        # # 查看某个 token 的词向量，例如 "苹果"
        # token = "苹果手机系统好"
        # token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        # # 如果 token 被拆分成多个子词，可以对它们取平均或者分别查看
        # for tid in token_ids:
        #     print(f"Token ID: {tid}, 词向量：", embeddings[tid])


if __name__ == '__main__':

    run_mon_train(types='test')












