import pandas as pd
import random

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


def load_stopwords(file_path):
    stopwords = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            stopwords.append(line.strip())
    return stopwords


def read_data_xls():
    stopwords = load_stopwords('./data/stopword')
    df = pd.read_excel('./data/category_train.xls')
    # data = df.apply(lambda x: x.astype(str).apply(lambda word: ' '.join(filter(lambda w: w not in stopwords, word.split()))))
    return df

