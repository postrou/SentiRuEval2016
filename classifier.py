import pandas as pd


def solving_rule(x):
    sentiment_sum = x.positive - x.negative
    if abs(sentiment_sum) > 0:
        sentiment_sum /= x.positive + x.negative
    if sentiment_sum < -0.05:
        return -1
    elif sentiment_sum > 0.05:
        return 1
    return 0


def classify(X, y):
    y_pred = []
    for index, x in X.iterrows():
        y_pred.append(solving_rule(x))
    # print('F1_micro =', f1_score(y, y_pred, average='micro'))
    # print('F1_macro =', f1_score(y, y_pred, average='macro'))
    return y_pred