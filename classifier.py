import pandas as pd


def solving_rule(x):
    """ Rule-based solving rule
    """
    sentiment_sum = x.positive - x.negative
    if abs(sentiment_sum) > 0:
        sentiment_sum /= x.positive + x.negative
    if sentiment_sum < -0.05:
        return -1
    elif sentiment_sum > 0.05:
        return 1
    return 0


def classify(X):
    """ Get class labels for X
    """
    y_pred = []
    for index, x in X.iterrows():
        y_pred.append(solving_rule(x))
    return y_pred