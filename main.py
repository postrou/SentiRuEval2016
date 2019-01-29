import os
import sys
import re

import pandas as pd
from sklearn.metrics import f1_score

from data_preprocessing import build_dataset, build_sentiment_dict
from classifier import classify


def main():
    data_dir = sys.argv[1]
    filenames = os.listdir(data_dir)

    rusentilex_path = sys.argv[2]
    sentiment_dict = {}
    build_sentiment_dict(sentiment_dict, rusentilex_path)

    datasets_banks = []
    datasets_tkk = []
    for fn in filenames:
        if re.fullmatch('bank.+(train|etalon).+', fn):
            dataset = build_dataset(os.path.join(data_dir, fn), sentiment_dict)
            datasets_banks.append(dataset)
        elif re.fullmatch('tkk.+(train|etalon).+', fn):
            dataset = build_dataset(os.path.join(data_dir, fn), sentiment_dict)
            datasets_tkk.append(dataset)
    datasets_banks_full = pd.concat(datasets_banks)
    datasets_tkk_full = pd.concat(datasets_tkk)

    X = datasets_banks_full.iloc[:, :-1]
    y = list(datasets_banks_full.iloc[:, -1])
    y_pred = classify(X, y)

    print('Banks')
    print('F1_micro =', f1_score(y, y_pred, average='micro'))
    print('F1_macro =', f1_score(y, y_pred, average='macro'), '\n')

    X = datasets_tkk_full.iloc[:, :-1]
    y = list(datasets_tkk_full.iloc[:, -1])
    y_pred = classify(X, y)

    print('TKK')
    print('F1_micro =', f1_score(y, y_pred, average='micro'))
    print('F1_macro =', f1_score(y, y_pred, average='macro'))

if __name__ == '__main__':
    main()
