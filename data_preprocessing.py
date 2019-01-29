from collections import defaultdict
import xml.etree.cElementTree as et
import re

import numpy as np
import pandas as pd
import pymorphy2
from nltk.tokenize import TweetTokenizer
from tqdm import trange


morph = pymorphy2.MorphAnalyzer()


def build_sentiment_dict(target_dict, rusentilex_path):
    """ Building word-sentiment dictionary, based on ReSentiLex 2017
    """
    possible_sentiments = ['positive', 'negative', 'neutral']
    with open(rusentilex_path, 'r') as f:
        for line in f.readlines():
            
            if line[0] == '!' or line == '\n':
                continue
                
            tokens = line.split(', ')
            sentiment = tokens[3]
            if sentiment in possible_sentiments:
                target_dict.update({tokens[0]: sentiment})


def build_dataset(file_path, sentiment_dict):
    """ Building dataset from file in file_path
    """
    root = et.parse(file_path).getroot()
    
    data = defaultdict(list)
    tables = root[-1]
    for i in trange(len(tables), desc='Building dataset {}'.format(file_path)):
        useful_info = False
        
        for column in tables[i]:
            name = column.attrib['name']
            
            if useful_info and column.text != 'NULL':
                for feature_name, feature_value in features.items():
                    data[feature_name].append(feature_value)
                value = int(column.text)
                data['sentiment'].append(value)
                    
            if name == 'text':
                features = preprocess_text(column.text, sentiment_dict)
                useful_info = True

    dataset = pd.DataFrame.from_dict(data)
    return dataset


def preprocess_text(text, sentiment_dict):
    """ Tweet text preprocessing
    """
    tokens = tokenize(text)
    n_of_positive = 0
    n_of_negative = 0
    for i in range(len(tokens)):
        token_sentiment = sentiment_dict.get(tokens[i])
        if token_sentiment == 'positive':
            if i > 0 and tokens[i - 1] == 'не':
                n_of_negative += 1
            else:
                n_of_positive += 1
            
        elif token_sentiment == 'negative':
            if i > 0 and tokens[i - 1] == 'не':
                n_of_positive += 1
            else:
                n_of_negative += 1
            
    return {'text': tokens, 'negative': n_of_negative, 'positive': n_of_positive}


def tokenize(text):
    """ Tweet text tokenizing
    """
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    
    tokens = tokenizer.tokenize(text)
    for i in range(len(tokens)):
        tokens[i] = re.sub('#+', '', tokens[i])
    
    tokens = [morph.parse(token)[0].normal_form for token in tokens if not token.startswith('http')]
    return tokens