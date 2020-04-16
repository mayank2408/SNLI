import pandas as pd
import spacy
from tqdm import tqdm

train_p = 'data/snli_1.0/snli_1.0_train.txt'
dev_p = 'data/snli_1.0/snli_1.0_dev.txt'
test_p= 'data/snli_1.0/snli_1.0_test.txt'

train_df = pd.read_csv(train_p, sep='\t', keep_default_na=False)
dev_df = pd.read_csv(dev_p, sep='\t', keep_default_na=False)
test_df = pd.read_csv(test_p, sep='\t', keep_default_na=False)

def convert(df):
    return list(zip(df['sentence1'], df['sentence2'], df['gold_label']))

train_list = convert(train_df)
dev_list = convert(dev_df)
test_list = convert(test_df)

def clean(data):
    return [(sent1, sent2, label) for (sent1, sent2, label) in data if label != '-']

train_data = clean(train_list)
dev_data = clean(dev_list)
test_data = clean(test_list)

token = spacy.load('en_core_web_sm')

def tokenize(string):
    return ' '.join([token.text for token in token.tokenizer(string)])


def tokenize_data(data):
    return [(tokenize(sent1), tokenize(sent2), label) for (sent1, sent2, label) in tqdm(data)]

train_data = tokenize_data(train_data)
dev_data = tokenize_data(dev_data)
test_data = tokenize_data(test_data)

train_df = pd.DataFrame.from_records(train_data)
dev_df = pd.DataFrame.from_records(dev_data)
test_df = pd.DataFrame.from_records(test_data)

headers = ['sentence1', 'sentence2', 'label']

train_df.to_csv(f'{train_p[:-4]}.csv', index=False, header=headers)
dev_df.to_csv(f'{dev_p:-4]}.csv', index=False, header=headers)
test_df.to_csv(f'{test_p[:-4]}.csv', index=False, header=headers)
