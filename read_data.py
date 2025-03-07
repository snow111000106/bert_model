import pandas as pd
import random
import os
import pickle
from gensim.models import KeyedVectors


cache_file_path = './data/word2vec_cache.pkl'
word2vec_path = './data/cc.zh.300.vec.gz'


def load_word2vec_model():
    # 如果缓存文件存在，则直接加载缓存
    if os.path.exists(cache_file_path):
        print("加载缓存的 Word2Vec 模型...")
        with open(cache_file_path, 'rb') as f:
            word2vec = pickle.load(f)
    else:
        # 否则从原始文件加载并缓存
        print("从原始文件加载 Word2Vec 模型...")
        word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
        # 将加载的模型保存到缓存
        with open(cache_file_path, 'wb') as f:
            pickle.dump(word2vec, f)

    return word2vec


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
    # data = df.apply(lambda x: x.astype(str).apply(
    # lambda word: ' '.join(filter(lambda w: w not in stopwords, word.split()))))
    return df

