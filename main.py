import sys

from sklearn.metrics import f1_score

from data_preprocessing import build_dataset, build_sentiment_dict
from classifier import classify


if __name__ == '__main__':
    file_path = sys.argv[1]
    rusentilex_path = sys.argv[2]

    sentiment_dict = {}
    build_sentiment_dict(sentiment_dict, rusentilex_path)

    dataset = build_dataset(file_path, sentiment_dict)
    X = dataset.iloc[:, :-1]
    y = list(dataset.iloc[:, -1])

    y_pred = classify(X, y)

    print('F1_micro =', f1_score(y, y_pred, average='micro'))
    print('F1_macro =', f1_score(y, y_pred, average='macro'))