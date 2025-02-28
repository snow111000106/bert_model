import torch
import numpy as np
from transformers import BertTokenizer
from config import BERT_PATH, labels_moon, category_label


tokenizer = BertTokenizer.from_pretrained(BERT_PATH)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df):

        self.labels = [category_label[str(label)] for label in df['label']]
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


class SentimentDataset(torch.utils.data.Dataset):

    def __init__(self, data, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = data['text'].values
        self.labels = data['label'].apply(lambda x: 1 if x == 'pos' else 0).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }